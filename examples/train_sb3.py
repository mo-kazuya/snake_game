"""Train an agent on gym_snake with Stable-Baselines3 (PPO).

This is an *optional* example — it needs extra dependencies:

    pip install "stable-baselines3>=2.0" torch

Then:

    # fast MLP training on the compact feature observation
    python examples/train_sb3.py --obs features --timesteps 200000

    # CNN training on the grid observation
    python examples/train_sb3.py --obs grid --timesteps 500000

The trained model is saved to ``ppo_snake.zip`` and a short greedy evaluation
is printed at the end.
"""

from __future__ import annotations

import argparse


def make_env_fn(grid: int, obs_type: str):
    import gymnasium as gym

    import gym_snake  # noqa: F401

    def _thunk():
        return gym.make("gym_snake/Snake-v0", grid_size=grid, obs_type=obs_type)

    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on gym_snake")
    parser.add_argument("--obs", choices=["features", "grid"], default="features")
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out", default="ppo_snake.zip")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        raise SystemExit(
            "This example needs stable-baselines3 and torch:\n"
            '    pip install "stable-baselines3>=2.0" torch'
        )

    policy = "MlpPolicy" if args.obs == "features" else "CnnPolicy"
    env = DummyVecEnv([make_env_fn(args.grid, args.obs) for _ in range(args.n_envs)])

    # CnnPolicy expects channel-first image obs, which SnakeEnv already provides.
    # These hyperparameters reproduce the run in examples/TRAINING_RESULTS.md
    # (10x10, features, 2M steps -> mean score ~23).
    model = PPO(
        policy, env, verbose=1,
        n_steps=512, batch_size=512,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"saved model to {args.out}")

    # Greedy evaluation.
    eval_env = make_env_fn(args.grid, args.obs)()
    scores = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=1000 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
        scores.append(info["score"])
    print(f"eval scores over 10 episodes: {scores} (mean {sum(scores) / len(scores):.1f})")
    eval_env.close()


if __name__ == "__main__":
    main()
