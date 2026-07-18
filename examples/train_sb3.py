"""Train an agent on gym_snake with Stable-Baselines3 (PPO).

This is an *optional* example — it needs extra dependencies:

    pip install "stable-baselines3>=2.0" torch

Then:

    # fast MLP training on the compact feature observation
    python examples/train_sb3.py --obs features --timesteps 200000

    # CNN on the egocentric observation (recommended CNN setup; the obs shape
    # is board-size independent, so the trained model runs on any grid)
    python examples/train_sb3.py --obs ego --timesteps 3000000

    # CNN on the raw full-board grid observation (weights tied to this --grid)
    python examples/train_sb3.py --obs grid --timesteps 3000000

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
    parser.add_argument("--obs", choices=["features", "grid", "ego"], default="features")
    parser.add_argument(
        "--arch", choices=["cnn", "transformer"], default="cnn",
        help="feature extractor for image observations (ego/grid); "
             "'transformer' is only supported with --obs ego",
    )
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
        # Both CNN setups use SmallGridCNN (stride-1; SB3's default NatureCNN
        # collapses on boards this small):
        #
        # * --obs ego (recommended): head-centered, heading-up rotated local
        #   view + minimap. Fixed (5, 11, 11) shape -> the trained model runs
        #   on ANY board size, and it learns far faster than raw grid pixels
        #   (mean ~45 on 10x10 in 3M steps vs ~14.5 for raw grid).
        # * --obs grid: raw (3, H, W) board pixels; the flatten head ties the
        #   weights to this --grid size, and learning is slow (a constant-LR /
        #   low-entropy run plateaus at mean ~2; this tuned recipe reaches ~15).
        # * --arch transformer (--obs ego only): ViT-style EgoTransformer.
        #   Works, but plain PPO from scratch is far less sample-efficient than
        #   the CNN on this task (mean ~11 vs ~36 at 1.5M steps on 10x10).
        #   The strong recipe for this architecture is imitation learning from
        #   the search AI + PPO fine-tuning: see train_transformer_v2.py.
        if args.arch == "transformer":
            if args.obs != "ego":
                raise SystemExit("--arch transformer requires --obs ego")
            from gym_snake.policies import transformer_policy_kwargs

            policy_kwargs = transformer_policy_kwargs(
                d_model=64, num_layers=2, dim_feedforward=128, patch_size=2,
            )
        else:
            from gym_snake.policies import cnn_policy_kwargs

            policy_kwargs = cnn_policy_kwargs()

        def linear_decay(progress_remaining: float) -> float:
            return 3e-4 * progress_remaining

        model = PPO(
            "CnnPolicy", env, verbose=1,
            n_steps=1024, batch_size=2048, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
            learning_rate=linear_decay,
            policy_kwargs=policy_kwargs,
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
