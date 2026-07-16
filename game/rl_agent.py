"""Adapter that drives the Django Snake with the trained PPO agents.

Two reinforcement-learning models were trained in the ``gym_snake`` environment
(see ``examples/TRAINING_RESULTS.md``), both using **relative actions**
``0=straight, 1=turn-right, 2=turn-left``:

* ``"rl"``      — an MLP on the **11-dim ``features`` observation**. The feature
                  encoding is *independent of the board size*, so it runs on any
                  Django board (default 20x20).
* ``"rl_cnn"``  — a CNN on the **``(3, H, W)`` ``grid`` observation**. The CNN's
                  flatten->linear head is fixed to the ``10x10`` board it trained
                  on, so this model **only runs on a 10x10 board**.

This module converts a Django :class:`~game.engine.GameState` into the matching
observation, asks the policy for an action, and converts the relative action
back into an absolute direction the engine understands.

Everything degrades gracefully: if Stable-Baselines3 / PyTorch (or, for the CNN,
the ``gym_snake`` package that defines its feature extractor) aren't installed,
or a model file is missing, :func:`is_available` returns ``False`` for that
strategy and callers fall back to the search-based AI in :mod:`game.ai`.
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

_EXAMPLES = Path(settings.BASE_DIR) / "examples"

# Registry of trained models. ``grid`` is the board size the model requires;
# ``None`` means the observation is size-independent (runs on any board).
_MODELS = {
    "rl": {
        "file": _EXAMPLES / "ppo_snake_features.zip",
        "obs": "features",
        "grid": None,
        "label": "学習済みAI (features/MLP)",
        "needs_gym_snake": False,
    },
    "rl_cnn": {
        "file": _EXAMPLES / "ppo_snake_grid.zip",
        "obs": "grid",
        "grid": 10,
        "label": "学習済みAI (grid/CNN)",
        "needs_gym_snake": True,  # SmallGridCNN must be importable to load it
    },
}


# -- availability & metadata ----------------------------------------------


def is_available(name: str) -> bool:
    """True if model ``name`` can actually be used (deps + file present)."""
    cfg = _MODELS.get(name)
    if cfg is None or not cfg["file"].exists():
        return False
    try:
        import stable_baselines3  # noqa: F401

        if cfg["needs_gym_snake"]:
            import gym_snake.policies  # noqa: F401
    except Exception:
        return False
    return True


def required_grid(name: str):
    """Board size model ``name`` requires, or ``None`` if size-independent."""
    cfg = _MODELS.get(name)
    return cfg["grid"] if cfg else None


def strategies_meta() -> dict:
    """Per-model metadata for the frontend (availability + required board)."""
    return {
        name: {
            "label": cfg["label"],
            "grid": cfg["grid"],
            "available": is_available(name),
        }
        for name, cfg in _MODELS.items()
    }


# -- model loading (thread-safe, per model) -------------------------------

_load_lock = threading.Lock()
_models: dict[str, object] = {}


def _load_model(name: str):
    """Return the PPO policy for ``name``, deserializing it once (thread-safe)."""
    model = _models.get(name)
    if model is not None:
        return model
    with _load_lock:
        if name not in _models:
            from stable_baselines3 import PPO

            cfg = _MODELS[name]
            if cfg["needs_gym_snake"]:
                # Importing registers the custom SmallGridCNN class that the
                # saved CNN model references at load time.
                import gym_snake.policies  # noqa: F401

            # Force CPU: no GPU needed.
            _models[name] = PPO.load(str(cfg["file"]), device="cpu")
    return _models[name]


def warmup_async(name: str) -> None:
    """Load model ``name`` in a background thread so the first move isn't slow.

    Importing PyTorch and deserializing the policy takes a few seconds the first
    time. Kicking it off when a game is created (rather than on the first
    ``/api/step/``) hides that latency. Safe to call repeatedly.
    """
    if not is_available(name) or name in _models:
        return
    threading.Thread(target=_load_model, args=(name,), daemon=True).start()


# -- observations ----------------------------------------------------------


def _heading_index(direction: str) -> int:
    return _HEADINGS.index(DIRECTIONS[direction])


def build_observation(state: GameState, obs_type: str):
    """Convert a Django game state into the requested gym_snake observation."""
    if obs_type == "grid":
        return _grid_observation(state)
    return _feature_observation(state)


def _feature_observation(state: GameState):
    """Recreate gym_snake's 11-feature observation from a Django game state.

    Byte-for-byte consistent with ``SnakeEnv._feature_obs``.
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


def _grid_observation(state: GameState):
    """Recreate gym_snake's (3, H, W) grid observation from a Django state.

    Byte-for-byte consistent with ``SnakeEnv._grid_obs``:
    channel 0 = body, channel 1 = head, channel 2 = food.
    """
    import numpy as np

    g = state.grid
    obs = np.zeros((3, g, g), dtype=np.float32)
    for (x, y) in state.snake:
        obs[0, y, x] = 1.0
    hx, hy = state.snake[0]
    obs[1, hy, hx] = 1.0
    fx, fy = state.food
    obs[2, fy, fx] = 1.0
    return obs


# -- action selection ------------------------------------------------------


def choose_direction(state: GameState, name: str = "rl") -> str:
    """Return the absolute direction model ``name`` would take."""
    model = _load_model(name)
    obs = build_observation(state, _MODELS[name]["obs"])
    action, _ = model.predict(obs, deterministic=True)

    hidx = _heading_index(state.direction)
    a = int(action)
    if a == 1:            # turn right (clockwise)
        hidx = (hidx + 1) % 4
    elif a == 2:          # turn left (counter-clockwise)
        hidx = (hidx - 1) % 4
    return _DIR_NAMES[hidx]
