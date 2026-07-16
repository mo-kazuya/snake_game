"""Tests for the gym_snake environment, including Gymnasium's API checker."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_snake  # noqa: F401  (registers the env)
from gym_snake.envs import SnakeEnv


def test_registered_make():
    env = gym.make("gym_snake/Snake-v0", grid_size=10)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_check_env_grid():
    check_env(SnakeEnv(grid_size=8, obs_type="grid"), skip_render_check=True)


def test_check_env_features():
    check_env(SnakeEnv(grid_size=8, obs_type="features"), skip_render_check=True)


def test_reset_is_deterministic_with_seed():
    a = SnakeEnv(grid_size=10)
    b = SnakeEnv(grid_size=10)
    obs_a, _ = a.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_step_returns_five_tuple():
    env = SnakeEnv(grid_size=8)
    env.reset(seed=0)
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "score" in info


def test_eating_food_rewards_and_grows():
    env = SnakeEnv(grid_size=8, reward_shaping=False)
    env.reset(seed=0)
    # Place food directly ahead of the head (heading starts as 'right').
    hx, hy = env.snake[0]
    env.food = (hx + 1, hy)
    length_before = len(env.snake)
    _, reward, _, _, info = env.step(0)  # straight
    # Eating pays REWARD_FOOD, minus the per-step time penalty.
    assert reward == env.REWARD_FOOD + env.REWARD_STEP
    assert len(env.snake) == length_before + 1
    assert info["score"] == 1


def test_wall_collision_terminates():
    env = SnakeEnv(grid_size=6)
    env.reset(seed=0)
    # Drive straight right into the wall.
    terminated = False
    for _ in range(env.grid_size + 2):
        _, reward, terminated, _, info = env.step(0)
        if terminated:
            break
    assert terminated
    assert info["death"] == "wall"
    assert reward <= env.REWARD_DEATH + env.REWARD_STEP + 1e-6


def test_relative_turns_change_heading():
    env = SnakeEnv(grid_size=10)
    env.reset(seed=0)
    assert env.heading_idx == 1  # right
    env.step(1)  # turn right (clockwise) -> down
    assert env.heading_idx == 2
    env.step(2)  # turn left (ccw) -> right
    assert env.heading_idx == 1


def test_truncation_on_starvation():
    env = SnakeEnv(grid_size=8, max_steps_without_food=5, reward_shaping=False)
    env.reset(seed=0)
    env.food = (0, 0)  # far, and we won't reach it going in circles
    truncated = False
    for _ in range(20):
        # circle: right, down, left, up ...
        _, _, terminated, truncated, _ = env.step(1)
        if terminated or truncated:
            break
    assert truncated or terminated


def test_ansi_render():
    env = SnakeEnv(grid_size=6, render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "@" in out  # the head marker


def test_rgb_render_shape():
    env = SnakeEnv(grid_size=6, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
