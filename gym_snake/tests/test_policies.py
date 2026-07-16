"""Tests for the custom CNN feature extractor.

Skipped automatically when PyTorch / Stable-Baselines3 (the ``[train]`` extra)
are not installed.
"""

from __future__ import annotations

import pytest

from gym_snake.envs import SnakeEnv

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gym_snake.policies import SmallGridCNN  # noqa: E402


def test_extractor_output_shape():
    env = SnakeEnv(grid_size=10, obs_type="grid")
    extractor = SmallGridCNN(env.observation_space, features_dim=256)
    obs, _ = env.reset(seed=0)
    batch = torch.as_tensor(obs).unsqueeze(0)  # (1, 3, 10, 10)
    out = extractor(batch)
    assert out.shape == (1, 256)


def test_extractor_handles_various_board_sizes():
    for grid in (8, 12, 20):
        env = SnakeEnv(grid_size=grid, obs_type="grid")
        extractor = SmallGridCNN(env.observation_space, features_dim=128)
        obs, _ = env.reset(seed=0)
        out = extractor(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 128)
