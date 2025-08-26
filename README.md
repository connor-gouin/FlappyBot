# Flappy Bird RL (PPO)

Train a policy with **Proximal Policy Optimization (PPO)** to play a lightweight, custom **Flappy Bird** environment built on **Gymnasium** + **Stable-Baselines3**. Includes live rendering, TensorBoard logging, and **dynamic difficulty** that tightens the course as you pass pipes.

---

## ✨ Features

- **Custom env (`envs/flappy_env.py`)**
  - Vector observations for speed/stability
  - Correct physics (up = +y; **gravity < 0**)
  - **Dynamic difficulty**: every _N_ pipes, the gap narrows and the spawn interval shortens (resets every episode)
  - (Optional) **Speed growth** so newly spawned pipes move faster at higher difficulty
  - Reward shaping: small alive bonus, big pass bonus, “center-the-gap” shaping, tiny flap cost
  - Lightweight **matplotlib** renderer; optional GIF export during eval
- **Training (`train.py`)**
  - PPO with vectorized envs (Dummy/Subproc)
  - TensorBoard logging out of the box
- **Evaluation (`evaluate.py`)**
  - Deterministic or stochastic rollouts
  - `--render` for live viewer
  - `--save_gif` to record the first episode

---

## 🧱 Project Structure

```
.
├─ envs/
│  └─ flappy_env.py          # Custom Gymnasium environment
├─ train.py                  # PPO training entrypoint
├─ evaluate.py               # Evaluation / render / GIF
├─ models/                   # Saved models (.zip from SB3)
├─ tb/                       # TensorBoard logs
└─ README.md
```

---

## 🛠️ Setup

> On Windows, consider placing this project in a **non-OneDrive** folder to avoid file-locking pauses during logging/checkpoints.

### 1) Create & activate a virtual environment

```bash
# Windows (PowerShell)
py -3 -m venv .venv
. .venv/Scripts/Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install gymnasium==1.2.0 stable-baselines3==2.7.0 torch numpy matplotlib imageio
```

---

## 🚀 Quickstart

### Train

```bash
# Default (per your train.py): ~1,000,000 timesteps, logs to tb/
python train.py
```

Common flags (if your `train.py` uses argparse):

```bash
# Easier curriculum + more envs for throughput
python train.py --easy --num_envs 12 --total_timesteps 300000

# Go long
python train.py --num_envs 12 --total_timesteps 5000000
```

### TensorBoard

```bash
# If 'tensorboard' is on PATH:
tensorboard --logdir tb --host 127.0.0.1 --port 6010

# If not, on Windows:
python -m tensorboard.main --logdir tb --host 127.0.0.1 --port 6010
```

Open the URL (e.g., http://127.0.0.1:6010) → **Scalars** → tick your run (e.g., `ppo_flappy_*`).

Key curves:
- `rollout/ep_rew_mean` (learning signal)
- `rollout/ep_len_mean` (survival time)
- `train/policy_entropy`, `train/clip_fraction` (health of updates)

### Evaluate (render / GIF)

```bash
# Greedy (no sampling) with live viewer
python evaluate.py --deterministic --render

# Save a GIF of the first episode (requires imageio)
python evaluate.py --deterministic --save_gif run.gif

# If you trained/evaluated with easier settings
python evaluate.py --easy --deterministic --render
```

---

## 🧪 Environment Details

### Observation (shape = 5)
```
[ bird_y, bird_vel, next_pipe_x, next_pipe_top, next_pipe_bottom ]
```
- `bird_y`           — normalized vertical position `[0..1]`
- `bird_vel`         — vertical velocity (clamped)
- `next_pipe_x`      — distance from the bird (x=0.2) to next pipe
- `next_pipe_top`    — top of gap
- `next_pipe_bottom` — bottom of gap

### Action (Discrete 2)
- `0` = do nothing
- `1` = flap

> By default the flap **sets** upward velocity to a small impulse (e.g., `self.bird_vel = flap_impulse`).  
> You can switch to an **additive** impulse (`self.bird_vel += flap_impulse`) for smoother control; keep the `max_vel` clamp either way.

### Reward Shaping (defaults)
- `+0.02` per step alive
- `+3.0` for passing a pipe
- Up to `+0.5` bonus when near the **gap center** as the pipe approaches
- `−0.002` per flap (discourage spam)
- `−1.0` on death (`out_of_bounds` or `pipe_hit`)
- **Time limit** ends episodes with `truncated=True` (not a failure)

### Dynamic Difficulty
- Every `diff_every_pipes` (e.g., 10) pipes passed:
  - `pipe_gap ← max(min_pipe_gap, pipe_gap * gap_decay)`
  - `pipe_interval ← max(min_pipe_interval, pipe_interval * interval_decay)`
  - (Optional) `pipe_speed ← min(max_pipe_speed, pipe_speed * speed_growth)`
- Difficulty resets to **base values** on every `reset()` so each episode starts fair.

---

## 🔧 Key Tunables (in `flappy_env.py`)

```python
gravity       = -0.0022   # pulls down
flap_impulse  = 0.014     # upward kick (set-to by default)
max_vel       = 0.05      # vertical speed clamp

pipe_gap      = 0.28
pipe_speed    = 0.006
pipe_interval = 120
max_steps     = 20000

# Dynamic difficulty
dynamic_difficulty = True
diff_every_pipes   = 10
gap_decay          = 0.85
interval_decay     = 0.85
min_pipe_gap       = 0.10
min_pipe_interval  = 20

# Optional speed growth
# speed_growth = 1.05
# max_pipe_speed = 0.012
# accelerate_existing_pipes = False  # only new pipes speed up by default
```

---

## 🧠 PPO Configuration (suggested)

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,                        # vectorized env (Dummy/Subproc)
    verbose=1,
    n_steps=1024,
    batch_size=2048,
    gamma=0.995,
    gae_lambda=0.95,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.02,              # encourage exploration; tune 0.01–0.03
    tensorboard_log="tb/"
    # policy_kwargs=dict(net_arch=[256,256])  # optional larger net
)
```

**Deterministic vs stochastic**
- Use `--deterministic` for final eval (greedy actions).
- Without it, actions are sampled (adds noise; good for robustness checks).

---

## 🧭 Tips & Curriculum

- **Break the 1-pipe plateau**: increase pass reward (`+3`→`+4`), reduce alive bonus (`0.02`→`0.01`), keep the gap-centering bonus, try `ent_coef=0.02–0.03`.
- **Curriculum**: start easier (`pipe_gap=0.30`, `pipe_speed=0.006`), train until avg ≥ 3–5 pipes, then harden settings.
- **Parallelism**: use `SubprocVecEnv` with 8–16 envs if your CPU can handle it.
- **Seeds**: you can train with one seed and evaluate with another; fix seeds for reproducible evals.
