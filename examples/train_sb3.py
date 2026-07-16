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

    env = DummyVecEnv([make_env_fn(args.grid, args.obs) for _ in range(args.n_envs)])

    # These hyperparameters reproduce the runs in examples/TRAINING_RESULTS.md.
    if args.obs == "features":
        # The compact feature MLP learns fast with a constant LR.
        model = PPO(
            "MlpPolicy", env, verbose=1,
            n_steps=512, batch_size=512,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
        )
    else:
        # SB3's default NatureCNN can't handle a 10x10 board; use a small,
        # stride-1 CNN that preserves the board resolution.
        #
        # Learning from raw pixels is far less sample-efficient than from the
        # hand-crafted features. A constant-LR / low-entropy run plateaus around
        # mean score ~2; more exploration (higher ent_coef), larger, less
        # frequently-updated rollouts, and a decaying learning rate break past
        # that plateau to mean score ~15 over 3M steps.
        from gym_snake.policies import cnn_policy_kwargs

        def linear_decay(progress_remaining: float) -> float:
            return 3e-4 * progress_remaining

        model = PPO(
            "CnnPolicy", env, verbose=1,
            n_steps=1024, batch_size=2048, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
            learning_rate=linear_decay,
            policy_kwargs=cnn_policy_kwargs(),
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
