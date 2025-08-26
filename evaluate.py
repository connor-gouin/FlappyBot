import argparse
from pathlib import Path
import time
from typing import Optional, List

from stable_baselines3 import PPO
from envs.flappy_env import FlappyEnv

# Optional: only needed if you use --save_gif
try:
    import imageio.v2 as imageio  # pip install imageio
except Exception:
    imageio = None


def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained PPO Flappy Bird agent")
    p.add_argument("--model", type=str, default=None,
                   help="Path to model .zip (.zip optional). If omitted, tries models/best/best_model.zip then models/ppo_flappy(.zip)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic policy (recommended for final eval).")
    p.add_argument("--seed", type=int, default=0)

    # Match training difficulty
    p.add_argument("--pipe_gap", type=float, default=0.22)
    p.add_argument("--pipe_speed", type=float, default=0.008)
    p.add_argument("--easy", action="store_true", help="Evaluate with wider gap/slower pipes (curriculum).")

    # Visualization / recording
    p.add_argument("--render", action="store_true", help="Live matplotlib viewer (requires env.render()).")
    p.add_argument("--save_gif", type=str, default=None,
                   help="Path to save a GIF of the FIRST episode (requires env.frame_rgb()). Example: run.gif")
    p.add_argument("--gif_fps", type=int, default=30)
    p.add_argument("--render_sleep", type=float, default=0.01, help="Delay between frames when rendering (seconds).")
    return p.parse_args()


def pick_model_path(cli_model: Optional[str]) -> str:
    if cli_model:
        # allow model path with or without .zip
        p = Path(cli_model)
        if p.suffix != ".zip" and p.with_suffix(".zip").exists():
            return str(p.with_suffix(".zip"))
        return str(p)
    candidates = [
        "models/best/best_model.zip",
        "models/ppo_flappy.zip",
        "models/ppo_flappy",  # in case it was saved without .zip
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("No model found. Pass --model or train a model first.")


def maybe_record_frame(env, frames: List):
    """
    Append an RGB frame if env provides frame_rgb(); otherwise no-op.
    """
    if frames is None:
        return
    if hasattr(env, "frame_rgb"):
        frame = env.frame_rgb()
        if frame is not None:
            frames.append(frame)


if __name__ == "__main__":
    args = parse_args()

    if args.easy:
        args.pipe_gap = 0.30
        args.pipe_speed = 0.006

    model_path = pick_model_path(args.model)
    print(f"== LOADING MODEL: {model_path}")

    env = FlappyEnv(seed=args.seed, pipe_gap=args.pipe_gap, pipe_speed=args.pipe_speed)
    model = PPO.load(model_path, print_system_info=True)

    total_reward = 0.0
    total_pipes = 0
    crash_reasons = {}

    save_gif_this_run = args.save_gif is not None
    if save_gif_this_run and imageio is None:
        print("WARNING: imageio not installed; GIF recording will be skipped. Run: pip install imageio")
        save_gif_this_run = False

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_r = 0.0
        done = False
        info = {}

        frames = [] if (save_gif_this_run and ep == 0) else None

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, done, trunc, info = env.step(int(action))
            ep_r += r

            # Live view
            if args.render and hasattr(env, "render"):
                env.render()
                if args.render_sleep > 0:
                    time.sleep(args.render_sleep)

            # Recording
            maybe_record_frame(env, frames)

        # End of episode
        total_reward += ep_r
        total_pipes += env.score
        reason = info.get("reason", "unknown")
        crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
        print(f"Episode {ep+1}: reward={ep_r:.2f}, pipes_passed={env.score}, end_reason={reason}")

        # Save GIF for the first episode if requested
        if ep == 0 and frames and save_gif_this_run:
            try:
                imageio.mimsave(args.save_gif, frames, fps=args.gif_fps)
                print(f"Saved GIF to {args.save_gif} ({len(frames)} frames @ {args.gif_fps} fps)")
            except Exception as e:
                print(f"Failed to save GIF: {e}")

    avg_r = total_reward / args.episodes
    avg_pipes = total_pipes / args.episodes
    print(f"\nAvg reward over {args.episodes} episodes: {avg_r:.2f}")
    print(f"Avg pipes over {args.episodes} episodes:  {avg_pipes:.2f}")
    print(f"Crash reasons: {crash_reasons}")
