"""Custom CNN feature extractor for the grid observation of gym_snake.

Stable-Baselines3's default ``CnnPolicy`` uses ``NatureCNN`` (8x8/4x4 kernels
with large strides), which is designed for Atari-sized ~84x84 frames and
collapses to nothing on a 10x10 Snake board. This module provides a small,
stride-1 CNN that keeps the spatial resolution and works on tiny grids.

Because a saved SB3 model stores a reference to its ``features_extractor_class``,
this class lives in the importable ``gym_snake`` package so trained CNN models
can be reloaded anywhere with ``PPO.load(...)``.

Requires PyTorch (installed via the ``[train]`` extra).
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallGridCNN(BaseFeaturesExtractor):
    """A compact CNN suitable for small (H, W) Snake boards.

    Two stride-1 convolutions (padding preserves the board size) followed by a
    linear projection to ``features_dim``. The input is a ``(3, H, W)`` float
    image with channels ``[body, head, food]`` already in ``[0, 1]`` — so the
    policy must be created with ``normalize_images=False``.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Infer the flattened size from a dummy forward pass.
        with torch.no_grad():
            sample = torch.zeros((1, *observation_space.shape), dtype=torch.float32)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def cnn_policy_kwargs(features_dim: int = 256) -> dict:
    """policy_kwargs for a PPO ``CnnPolicy`` using :class:`SmallGridCNN`."""
    return dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        normalize_images=False,  # obs is already 0/1, not 0-255
    )
