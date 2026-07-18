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


class AnyGridCNN(BaseFeaturesExtractor):
    """Board-size-independent CNN for the raw grid observation.

    .. warning::
        Kept for reference — in practice this extractor **failed to train** on
        Snake (mean score flatlined near 0 after 1M PPO steps). The global
        pooling destroys single-pixel relative geometry (head-vs-neck
        orientation, adjacent danger) that the policy needs. The working
        size-independent approach is the egocentric observation
        (``obs_type="ego"``, see :mod:`gym_snake.obs`), whose **fixed-shape
        observation** makes a plain flatten CNN size-independent instead.

    :class:`SmallGridCNN` flattens the ``64 x H x W`` feature map into a linear
    layer, tying the weights to the one board size it was trained on. This
    extractor removes that dependency:

    * **CoordConv** — two normalized coordinate channels (x, y in ``[-1, 1]``)
      are appended to the input so absolute positions survive pooling,
    * stride-1 convolutions preserve the board resolution,
    * **adaptive pooling** collapses the feature map to a fixed ``4 x 4``
      (average and max, concatenated), so the linear head always sees the same
      shape regardless of H and W.

    One set of weights therefore runs on 10x10 and 20x20 boards alike. As with
    :class:`SmallGridCNN`, create the policy with ``normalize_images=False``.
    """

    POOL = 4  # pooled spatial size, fixed for any board

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] + 2  # + coord channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        n_flatten = 64 * self.POOL * self.POOL * 2  # avg + max pooled maps
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, _, h, w = observations.shape
        device = observations.device
        ys = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        z = self.cnn(torch.cat([observations, xs, ys], dim=1))
        avg = nn.functional.adaptive_avg_pool2d(z, self.POOL)
        mx = nn.functional.adaptive_max_pool2d(z, self.POOL)
        return self.linear(torch.cat([avg, mx], dim=1).flatten(1))


def any_grid_policy_kwargs(features_dim: int = 256) -> dict:
    """policy_kwargs for a PPO ``CnnPolicy`` using :class:`AnyGridCNN`."""
    return dict(
        features_extractor_class=AnyGridCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        normalize_images=False,
    )


class EgoTransformer(BaseFeaturesExtractor):
    """ViT-style Transformer feature extractor for the ego observation.

    Each of the ``H x W`` cells of the ``(C, H, W)`` observation becomes one
    token (a linear projection of its C channel values) plus a learned
    positional embedding; a CLS token aggregates the board through
    ``num_layers`` pre-norm Transformer encoder blocks, and its final embedding
    is projected to ``features_dim``.

    Because the ego observation already has a fixed shape for every board
    size, this extractor — like the CNN — is board-size independent. Dropout
    is disabled by default: stochastic policies during PPO rollouts hurt more
    than they regularize.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        patch_size: int = 2,
    ):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        # Pad H/W up to a multiple of patch_size, then embed each patch with a
        # strided conv (the standard ViT patch embedding). patch_size=1 is the
        # per-cell tokenization; patch_size=2 quarters the token count, which
        # cuts the quadratic attention cost ~10x — important on CPU.
        self.patch_size = patch_size
        self._pad_h = (patch_size - h % patch_size) % patch_size
        self._pad_w = (patch_size - w % patch_size) % patch_size
        self.n_tokens = ((h + self._pad_h) // patch_size) * ((w + self._pad_w) // patch_size)

        self.embed = nn.Conv2d(c, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        x = observations
        if self._pad_h or self._pad_w:
            x = nn.functional.pad(x, (0, self._pad_w, 0, self._pad_h))
        # (B, C, H, W) -> (B, d_model, H/p, W/p) -> (B, tokens, d_model)
        x = self.embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def transformer_policy_kwargs(features_dim: int = 256, **kwargs) -> dict:
    """policy_kwargs for a PPO ``CnnPolicy`` using :class:`EgoTransformer`."""
    return dict(
        features_extractor_class=EgoTransformer,
        features_extractor_kwargs=dict(features_dim=features_dim, **kwargs),
        normalize_images=False,
    )


def load_ppo_for_grid(path, grid_size: int, device: str = "cpu", env=None):
    """Load a saved PPO model rebound to a ``grid_size`` x ``grid_size`` board.

    An SB3 checkpoint stores the observation space it was trained with, and
    ``model.predict`` rejects observations of any other shape. For a model whose
    feature extractor is size-independent (:class:`AnyGridCNN`), the weights are
    valid for every board size — only the stored space needs overriding, which
    is exactly what ``custom_objects`` does.
    """
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3 import PPO

    custom_objects = {
        "observation_space": spaces.Box(
            low=0.0, high=1.0, shape=(3, grid_size, grid_size), dtype=np.float32
        ),
        "action_space": spaces.Discrete(3),
    }
    return PPO.load(path, env=env, device=device, custom_objects=custom_objects)
