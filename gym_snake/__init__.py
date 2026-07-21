"""gym_snake: a Gymnasium environment for training Snake-playing agents.

Registering here means you can create the environment by id:

    import gymnasium as gym
    import gym_snake  # noqa: F401  (triggers registration)

    env = gym.make("gym_snake/Snake-v0", grid_size=12)          # single snake
    env = gym.make("gym_snake/SnakeBattle-v0", grid_size=12)    # 2-snake battle
"""

from gymnasium.envs.registration import register

from gym_snake.envs import SnakeBattleEnv, SnakeEnv

__all__ = ["SnakeEnv", "SnakeBattleEnv"]

register(
    id="gym_snake/Snake-v0",
    entry_point="gym_snake.envs:SnakeEnv",
    max_episode_steps=None,  # the env handles its own truncation
)

register(
    id="gym_snake/SnakeBattle-v0",
    entry_point="gym_snake.envs:SnakeBattleEnv",
    max_episode_steps=None,  # the env handles its own truncation
)
