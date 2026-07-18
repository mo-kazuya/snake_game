"""Run the gym_snake environment with a random policy.

This is the simplest possible smoke test / usage example:

    python examples/random_agent.py --episodes 3 --render
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import gym_snake  # noqa: F401  (registers gym_snake/Snake-v0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-agent rollout for gym_snake")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--render", action="store_true", help="print each frame (ansi)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    render_mode = "ansi" if args.render else None
    env = gym.make("gym_snake/Snake-v0", grid_size=args.grid, render_mode=render_mode)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if render_mode == "ansi":
                print(env.render())
                print()
        print(
            f"episode {ep + 1}: score={info['score']} "
            f"length={info['length']} steps={info['steps']} "
            f"return={total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
