"""Tests for the custom CNN feature extractor.

Skipped automatically when PyTorch / Stable-Baselines3 (the ``[train]`` extra)
are not installed.
"""

from __future__ import annotations

import pytest

from gym_snake.envs import SnakeEnv

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gym_snake.policies import AnyGridCNN, EgoTransformer, SmallGridCNN  # noqa: E402


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


def test_ego_transformer_wide_window_token_count():
    """A wider ego window means more tokens, no code change needed."""
    for window, patch, expected in ((11, 1, 121), (21, 1, 441), (21, 2, 121)):
        env = SnakeEnv(grid_size=10, obs_type="ego", ego_window=window)
        ext = EgoTransformer(env.observation_space, features_dim=64, d_model=32,
                             nhead=4, num_layers=1, dim_feedforward=64,
                             patch_size=patch)
        assert ext.n_tokens == expected
        assert ext.pos_embed.shape == (1, expected + 1, 32)
        obs, _ = env.reset(seed=0)
        assert ext(torch.as_tensor(obs).unsqueeze(0)).shape == (1, 64)


def test_ego_transformer_amp_is_a_cpu_noop():
    """amp=True must stay float32 (and identical) on CPU, so a GPU-trained
    model plays on the Django server exactly as it was evaluated."""
    env = SnakeEnv(grid_size=10, obs_type="ego", ego_window=21)
    plain = EgoTransformer(env.observation_space, features_dim=64, d_model=32,
                           nhead=4, num_layers=1, dim_feedforward=64)
    amped = EgoTransformer(env.observation_space, features_dim=64, d_model=32,
                           nhead=4, num_layers=1, dim_feedforward=64, amp=True)
    amped.load_state_dict(plain.state_dict())
    obs = torch.as_tensor(env.reset(seed=0)[0]).unsqueeze(0)
    with torch.no_grad():
        out = amped(obs)
    assert out.dtype == torch.float32
    assert torch.equal(out, plain(obs))
