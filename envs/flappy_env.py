import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FlappyEnv(gym.Env):
    """
    Flappy Bird-like environment (vector observations, fast & stable).

    Observation (5):
        [bird_y, bird_vel, next_pipe_x, next_pipe_top, next_pipe_bottom]

    Action (Discrete 2):
        0 = do nothing, 1 = flap (velocity impulse)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 gravity=-0.0022,          # up is +y; gravity pulls down
                 flap_impulse=0.014,       # upward kick (you can switch to additive if you prefer)
                 max_vel=0.05,             # clamp vertical velocity magnitude
                 pipe_gap=0.28,            # starting gap (normalized units)
                 pipe_speed=0.006,         # starting horizontal speed of pipes (leftward)
                 pipe_interval=120,        # steps between spawns (will change with difficulty)
                 max_steps=20000,
                 # --- Dynamic difficulty ---
                 dynamic_difficulty=True,  # tighten course as you score
                 diff_every_pipes=10,      # every N pipes passed, increase difficulty
                 gap_decay=0.85,           # multiply gap by this (<1 shrinks gap)
                 interval_decay=0.85,      # multiply interval by this (<1 spawns closer)
                 min_pipe_gap=0.10,        # lower bound on gap
                 min_pipe_interval=20,     # lower bound on interval (in steps)
                 # --- Speed growth controls (NEW) ---
                 speed_growth=1.05,        # multiply pipe speed by this each difficulty bump
                 max_pipe_speed=0.012,     # cap the pipe speed
                 accelerate_existing_pipes=False,  # if True, also speed up pipes already on screen
                 seed=None):
        super().__init__()

        # Physics
        self.gravity = float(gravity)
        self.flap_impulse = float(flap_impulse)
        self.max_vel = float(max_vel)

        # Game params (mutable with dynamic difficulty)
        self.pipe_gap = float(pipe_gap)
        self.pipe_speed = float(pipe_speed)          # "current" speed for NEW spawns
        self.pipe_interval = int(pipe_interval)
        self.max_steps = int(max_steps)

        # Keep BASE (starting) values so we can restore them each episode
        self._base_pipe_gap = float(pipe_gap)
        self._base_pipe_interval = int(pipe_interval)
        self._base_pipe_speed = float(pipe_speed)

        # Dynamic difficulty knobs
        self.dynamic_difficulty = bool(dynamic_difficulty)
        self.diff_every_pipes = int(diff_every_pipes)
        self.gap_decay = float(gap_decay)
        self.interval_decay = float(interval_decay)
        self.min_pipe_gap = float(min_pipe_gap)
        self.min_pipe_interval = int(min_pipe_interval)

        # Speed growth knobs (NEW)
        self.speed_growth = float(speed_growth)
        self.max_pipe_speed = float(max_pipe_speed)
        self.accelerate_existing_pipes = bool(accelerate_existing_pipes)

        # Spaces
        self.action_space = spaces.Discrete(2)
        high = np.array([1.0,  1.5, 1.0, 1.0, 1.0], dtype=np.float32)
        low  = np.array([0.0, -1.5, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # RNG
        self.rng = np.random.default_rng(seed)
        self.reset_seed = seed

        # World layout (normalized 0..1)
        self.screen_h = 1.0
        self.screen_w = 1.0
        self.pipe_width = 0.08

        # Render state
        self._fig = None
        self._ax = None
        self._bird = None
        self._pipe_patches = []

        self._reset_internal()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _reset_internal(self):
        # Restore base difficulty EVERY EPISODE
        self.pipe_gap = self._base_pipe_gap
        self.pipe_interval = self._base_pipe_interval
        self.pipe_speed = self._base_pipe_speed

        self.bird_y = 0.5
        self.bird_vel = 0.0
        self.t = 0
        self.score = 0
        self.steps = 0

        # Pipes: each is dict {x, top, bottom, speed, _passed?}
        self.pipes = []

        # Variable spawn timing (so interval can change mid-episode)
        self._last_spawn_step = 0

        # Dynamic difficulty bookkeeping
        self.difficulty_level = 0
        self._next_diff_score = self.diff_every_pipes

        self._spawn_pipe(initial=True)

    def _spawn_pipe(self, initial=False):
        center = float(self.rng.uniform(0.3, 0.7))
        top = center - self.pipe_gap / 2.0
        bottom = center + self.pipe_gap / 2.0
        # Initial pipe uses same x as others; spacing handled by scheduler
        x = 1.2
        # NEW: store per-pipe speed so existing pipes don't retroactively speed up
        self.pipes.append({"x": x, "top": top, "bottom": bottom, "speed": self.pipe_speed})

    def _next_pipe(self):
        bird_x = 0.2
        future = [p for p in self.pipes if p["x"] + self.pipe_width/2 >= bird_x]
        return min(future, key=lambda p: p["x"]) if future else None

    def _obs(self):
        p = self._next_pipe()
        if p is None:
            p = {"x": 1.0, "top": 0.3, "bottom": 0.7}
        bird_x = 0.2
        next_pipe_x = np.clip(p["x"] - bird_x, 0.0, 1.0)
        obs = np.array([
            np.clip(self.bird_y, 0.0, 1.0),
            float(np.clip(self.bird_vel, -1.5, 1.5)),
            next_pipe_x,
            np.clip(p["top"], 0.0, 1.0),
            np.clip(p["bottom"], 0.0, 1.0)
        ], dtype=np.float32)
        return obs

    def _increase_difficulty(self):
        """Tighten gap, spawn closer, and (optionally) speed up pipes."""
        self.difficulty_level += 1

        # Narrow the gap
        new_gap = max(self.min_pipe_gap, self.pipe_gap * self.gap_decay)
        # Shorten spacing (smaller interval => pipes closer in time)
        new_interval = max(self.min_pipe_interval, int(self.pipe_interval * self.interval_decay))
        # Speed up NEW pipes (capped)
        new_speed = min(self.max_pipe_speed, self.pipe_speed * self.speed_growth)

        self.pipe_gap = float(new_gap)
        self.pipe_interval = int(new_interval)
        self.pipe_speed = float(new_speed)

        # Optionally speed up pipes already on screen
        if self.accelerate_existing_pipes:
            for p in self.pipes:
                p["speed"] = self.pipe_speed

        # Optional debug:
        # print(f"[DD] lvl={self.difficulty_level} gap={self.pipe_gap:.3f} "
        #       f"interval={self.pipe_interval} speed={self.pipe_speed:.4f}")

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._obs(), {}

    def render(self):
        """Lightweight matplotlib viewer."""
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.set_aspect("equal")
            self._ax.set_title(f"Pipes passed: {self.score}")
            self._bird = self._ax.plot([0.2], [self.bird_y], marker="o")[0]
            self._pipe_patches = []
            plt.ion()
            plt.show()

        # Clear previous pipe patches
        for patch in self._pipe_patches:
            patch.remove()
        self._pipe_patches = []

        # Draw pipes
        for p in self.pipes:
            x_left = p["x"] - self.pipe_width / 2.0
            top_rect = Rectangle((x_left, 0), self.pipe_width, p["top"], alpha=0.3)
            bot_rect = Rectangle((x_left, p["bottom"]), self.pipe_width, 1 - p["bottom"], alpha=0.3)
            self._pipe_patches.extend([top_rect, bot_rect])
            self._ax.add_patch(top_rect)
            self._ax.add_patch(bot_rect)

        # Update bird
        self._bird.set_data([0.2], [self.bird_y])
        self._ax.set_title(f"Pipes passed: {self.score}")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def step(self, action):
        self.steps += 1

        # Base reward (alive)
        reward = 0.02

        # Action: (your current choice) set velocity to impulse value
        # If you'd rather additive, change to: self.bird_vel += self.flap_impulse
        if action == 1:
            self.bird_vel = self.flap_impulse
            reward -= 0.002  # tiny cost to discourage flap spam

        # Physics: gravity, clamp, integrate
        self.bird_vel += self.gravity
        self.bird_vel = float(np.clip(self.bird_vel, -self.max_vel, self.max_vel))
        self.bird_y += self.bird_vel

        # Move pipes (use per-pipe speeds so existing pipes keep their original speed)
        for p in self.pipes:
            p["x"] -= p.get("speed", self.pipe_speed)

        # Spawn new pipes based on elapsed steps since last spawn
        if (self.steps - self._last_spawn_step) >= self.pipe_interval:
            self._spawn_pipe()
            self._last_spawn_step = self.steps

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p["x"] + self.pipe_width > -0.2]

        done = False
        truncated = False
        info = {}
        bird_x = 0.2

        # Passing reward & difficulty bumps
        for p in self.pipes:
            passed = (p.get("_passed") is not True) and (p["x"] + self.pipe_width / 2.0 < bird_x)
            if passed:
                p["_passed"] = True
                reward += 3.0
                self.score += 1

                # Increase difficulty every N pipes (applies to future spawns)
                if self.dynamic_difficulty and self.score >= self._next_diff_score:
                    self._increase_difficulty()
                    self._next_diff_score += self.diff_every_pipes

        # Near-gap shaping bonus (only when pipe is close ahead)
        p = self._next_pipe()
        if p is not None:
            if (p["x"] - bird_x) < 0.2:
                gap_center = 0.5 * (p["top"] + p["bottom"])
                dist = abs(self.bird_y - gap_center)
                bonus = 0.5 * (1.0 - np.clip(dist / (self.pipe_gap * 0.5), 0.0, 1.0))
                reward += float(bonus)

        # Terminations
        if self.bird_y < 0.0 or self.bird_y > 1.0:
            reward -= 1.0
            done = True
            info["reason"] = "out_of_bounds"

        if not done and p is not None:
            within_x = (p["x"] - self.pipe_width / 2.0) <= bird_x <= (p["x"] + self.pipe_width / 2.0)
            if within_x and not (p["top"] <= self.bird_y <= p["bottom"]):
                reward -= 1.0
                done = True
                info["reason"] = "pipe_hit"

        # Time limit -> truncated (not a failure)
        if not done and self.steps >= self.max_steps:
            truncated = True
            info["reason"] = "time_limit"

        return self._obs(), float(reward), done, truncated, info
