"""Adapter that drives the Django Snake with the trained PPO agent.

The reinforcement-learning model was trained in the ``gym_snake`` environment
(see ``examples/TRAINING_RESULTS.md``) with:

* the **11-dim ``features`` observation** (danger sensors + heading one-hot +
  food direction), and
* **relative actions** ``0=straight, 1=turn-right, 2=turn-left``.

This module converts a Django :class:`~game.engine.GameState` into that exact
observation, asks the policy for an action, and converts the relative action
back into an absolute direction the engine understands.

Because the feature observation is **independent of the board size**, the model
trained on a 10x10 board transfers directly to the Django default 20x20 board.

Everything here degrades gracefully: if Stable-Baselines3 / PyTorch aren't
installed, or the model file is missing, :func:`is_available` returns ``False``
and callers fall back to the search-based AI in :mod:`game.ai`.
"""

from __future__ import annotations

import threading
from pathlib import Path

from django.conf import settings

from .engine import DIRECTIONS, GameState

# Headings ordered clockwise, matching gym_snake's SnakeEnv._HEADINGS.
#   index: 0=up  1=right  2=down  3=left
_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_DIR_NAMES = ["up", "right", "down", "left"]

MODEL_PATH = Path(settings.BASE_DIR) / "examples" / "ppo_snake_features.zip"


def is_available() -> bool:
    """True if the RL policy can actually be used (deps + model file present)."""
    if not MODEL_PATH.exists():
        return False
    try:
        import stable_baselines3  # noqa: F401
    except Exception:
        return False
    return True


_load_lock = threading.Lock()
_model = None


def _load_model():
    """Return the PPO policy, deserializing it exactly once (thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:  # double-checked: another thread may have loaded it
            from stable_baselines3 import PPO

            # Force CPU: no GPU needed and the model is tiny.
            _model = PPO.load(str(MODEL_PATH), device="cpu")
    return _model


def warmup_async() -> None:
    """Load the model in a background thread so the first move isn't slow.

    Importing PyTorch and deserializing the policy takes a few seconds the very
    first time. Kicking it off when a game is created (rather than on the first
    ``/api/step/``) hides that latency. Safe to call repeatedly; it only ever
    loads once.
    """
    if not is_available() or _model is not None:
        return
    threading.Thread(target=_load_model, daemon=True).start()


def _heading_index(direction: str) -> int:
    return _HEADINGS.index(DIRECTIONS[direction])


def build_observation(state: GameState):
    """Recreate gym_snake's 11-feature observation from a Django game state.

    Must stay byte-for-byte consistent with
    ``gym_snake.envs.snake_env.SnakeEnv._feature_obs``.
    """
    import numpy as np

    head = state.snake[0]
    hidx = _heading_index(state.direction)
    heading = _HEADINGS[hidx]
    right = _HEADINGS[(hidx + 1) % 4]
    left = _HEADINGS[(hidx - 1) % 4]
    body = set(state.snake[:-1])  # tail cell frees up next tick
    grid = state.grid

    def danger(vec) -> float:
        nx, ny = head[0] + vec[0], head[1] + vec[1]
        if not (0 <= nx < grid and 0 <= ny < grid):
            return 1.0
        return 1.0 if (nx, ny) in body else 0.0

    food = state.food
    feats = [
        danger(heading),                 # danger straight
        danger(right),                   # danger right
        danger(left),                    # danger left
        float(heading == _HEADINGS[0]),  # heading up
        float(heading == _HEADINGS[1]),  # heading right
        float(heading == _HEADINGS[2]),  # heading down
        float(heading == _HEADINGS[3]),  # heading left
        float(food[0] < head[0]),        # food is left
        float(food[0] > head[0]),        # food is right
        float(food[1] < head[1]),        # food is up
        float(food[1] > head[1]),        # food is down
    ]
    return np.asarray(feats, dtype=np.float32)


def choose_direction(state: GameState) -> str:
    """Return the absolute direction the PPO policy would take."""
    model = _load_model()
    obs = build_observation(state)
    action, _ = model.predict(obs, deterministic=True)

    hidx = _heading_index(state.direction)
    a = int(action)
    if a == 1:            # turn right (clockwise)
        hidx = (hidx + 1) % 4
    elif a == 2:          # turn left (counter-clockwise)
        hidx = (hidx - 1) % 4
    return _DIR_NAMES[hidx]
