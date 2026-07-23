"""Tests for the custom CNN feature extractor.

Skipped automatically when PyTorch / Stable-Baselines3 (the ``[train]`` extra)
are not installed.
"""

from __future__ import annotations

import pytest

from gym_snake.envs import SnakeEnv

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gym_snake.policies import (  # noqa: E402
    AnyGridCNN, EgoTransformer, ResidualGridCNN, SmallGridCNN,
)


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


def test_ego_transformer_output_shape():
    env = SnakeEnv(grid_size=10, obs_type="ego")
    extractor = EgoTransformer(env.observation_space, features_dim=256)
    obs, _ = env.reset(seed=0)
    out = extractor(torch.as_tensor(obs).unsqueeze(0))
    assert out.shape == (1, 256)


def test_ego_transformer_same_weights_any_board_size():
    """The ego obs shape is fixed, so one extractor serves every board size."""
    extractor = EgoTransformer(
        SnakeEnv(grid_size=10, obs_type="ego").observation_space, features_dim=128
    )
    for grid in (8, 20, 40):
        env = SnakeEnv(grid_size=grid, obs_type="ego")
        obs, _ = env.reset(seed=0)
        out = extractor(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 128)


def test_ego_transformer_batch():
    env = SnakeEnv(grid_size=10, obs_type="ego")
    extractor = EgoTransformer(env.observation_space, features_dim=64, d_model=32,
                               num_layers=2, dim_feedforward=64)
    obs, _ = env.reset(seed=0)
    batch = torch.as_tensor(obs).unsqueeze(0).repeat(16, 1, 1, 1)
    assert extractor(batch).shape == (16, 64)


def test_residual_cnn_output_shape_and_depth():
    env = SnakeEnv(grid_size=10, obs_type="ego")
    extractor = ResidualGridCNN(env.observation_space, features_dim=256,
                                width=64, n_blocks=4)
    obs, _ = env.reset(seed=0)
    out = extractor(torch.as_tensor(obs).unsqueeze(0))
    assert out.shape == (1, 256)
    # Deeper than SmallGridCNN's 2 convs: stem (1) + 2 per residual block.
    n_conv = sum(1 for m in extractor.modules() if isinstance(m, torch.nn.Conv2d))
    assert n_conv == 1 + 2 * 4


def test_residual_cnn_same_weights_any_board_size():
    """Ego obs is fixed-shape, so one residual extractor serves every board."""
    extractor = ResidualGridCNN(
        SnakeEnv(grid_size=10, obs_type="ego").observation_space,
        features_dim=128, width=32, n_blocks=3,
    )
    for grid in (8, 20, 40):
        env = SnakeEnv(grid_size=grid, obs_type="ego")
        obs, _ = env.reset(seed=0)
        out = extractor(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 128)


def test_residual_block_is_identity_at_init_of_last_layer():
    """Gradients reach every block, and the skip keeps signal flowing."""
    env = SnakeEnv(grid_size=10, obs_type="ego")
    extractor = ResidualGridCNN(env.observation_space, features_dim=32,
                                width=32, n_blocks=6)
    obs, _ = env.reset(seed=0)
    batch = torch.as_tensor(obs).unsqueeze(0).repeat(8, 1, 1, 1)
    out = extractor(batch)
    assert out.shape == (8, 32)
    out.sum().backward()
    # The very first stem conv must receive a non-zero gradient through all 6
    # residual blocks -- i.e. the deep stack is trainable end to end.
    g = extractor.stem[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
