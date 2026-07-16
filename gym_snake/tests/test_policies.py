"""Tests for the custom CNN feature extractor.

Skipped automatically when PyTorch / Stable-Baselines3 (the ``[train]`` extra)
are not installed.
"""

from __future__ import annotations

import pytest

from gym_snake.envs import SnakeEnv

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gym_snake.policies import AnyGridCNN, SmallGridCNN  # noqa: E402


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


def test_any_grid_cnn_output_shape_for_all_sizes():
    for grid in (8, 10, 16, 20):
        env = SnakeEnv(grid_size=grid, obs_type="grid")
        extractor = AnyGridCNN(env.observation_space, features_dim=256)
        obs, _ = env.reset(seed=0)
        out = extractor(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 256)


def test_any_grid_cnn_weights_transfer_across_sizes():
    """The same state_dict must be loadable for any board size."""
    env10 = SnakeEnv(grid_size=10, obs_type="grid")
    env20 = SnakeEnv(grid_size=20, obs_type="grid")
    a = AnyGridCNN(env10.observation_space, features_dim=256)
    b = AnyGridCNN(env20.observation_space, features_dim=256)
    b.load_state_dict(a.state_dict())  # must not raise

    # And the shared weights actually run on both sizes.
    for env, ext in ((env10, a), (env20, b)):
        obs, _ = env.reset(seed=0)
        out = ext(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 256)
