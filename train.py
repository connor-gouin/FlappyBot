import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs.flappy_env import FlappyEnv

def make_env(seed=42, pipe_gap=0.22, pipe_speed=0.008):
    def _init():
        env = FlappyEnv(seed=seed, pipe_gap=pipe_gap, pipe_speed=pipe_speed)
        return Monitor(env)
    return _init

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--easy", action="store_true",
                   help="Start with an easier task (wider gap / slower pipes).")
    return p.parse_args()

if __name__ == "__main__":
    # Keep each env from grabbing all CPU threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb", exist_ok=True)

    args = parse_args()

    # Task difficulty (curriculum starter)
    if args.easy:
        pipe_gap = 0.28   # wider gap
        pipe_speed = 0.006  # slower pipes
    else:
        pipe_gap = 0.22
        pipe_speed = 0.008

    # Vectorized envs across processes (better on multi-core CPUs)
    env_fns = [make_env(seed=args.seed + i, pipe_gap=pipe_gap, pipe_speed=pipe_speed)
               for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns, start_method="spawn")  # Windows-safe when guarded by __main__

    # Separate eval env (deterministic eval, single process)
    eval_env = make_env(seed=args.seed + 10, pipe_gap=pipe_gap, pipe_speed=pipe_speed)()
    run_name = f"ppo_flappy_{int(time.time())}"

    # Save the best model automatically
    stop_no_improve = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # stop if no improvement over N evals
        min_evals=10,
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="tb/eval",
        eval_freq=10_000,           # evaluate every N env steps
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_no_improve
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=2048,
        gae_lambda=0.95,
        gamma=0.995,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,             # encourage exploration
        tensorboard_log="tb/"
    )

    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name, callback=eval_cb)

    # Save final model (best model is also saved via EvalCallback)
    model.save("models/ppo_flappy")
    env.close()
    eval_env.close()
