"""Adapter that drives the Django Snake with the trained PPO agents.

Two reinforcement-learning models were trained in the ``gym_snake`` environment
(see ``examples/TRAINING_RESULTS.md``), both using **relative actions**
``0=straight, 1=turn-right, 2=turn-left``:

* ``"rl"``      — an MLP on the **11-dim ``features`` observation**. The feature
                  encoding is *independent of the board size*, so it runs on any
                  Django board (default 20x20).
* ``"rl_cnn"``  — a CNN on the **egocentric ``ego`` observation** (head-centered,
                  heading-up rotated local view + whole-board minimap, fixed
                  ``(5, 11, 11)`` shape). Because the observation shape never
                  depends on the board, this model **also runs on any board
                  size** — it was curriculum-trained on 10x10 then fine-tuned
                  on 20x20.

This module converts a Django :class:`~game.engine.GameState` into the matching
observation, asks the policy for an action, and converts the relative action
back into an absolute direction the engine understands.

Everything degrades gracefully: if Stable-Baselines3 / PyTorch (or, for the CNN,
the ``gym_snake`` package that defines its feature extractor) aren't installed,
or a model file is missing, :func:`is_available` returns ``False`` for that
strategy and callers fall back to the search-based AI in :mod:`game.ai`.

Most of these models were trained single-snake, so they can't be given real
opponent awareness without retraining. When a game has more than one snake, the
other living snake's body cells are folded into the *existing* danger/body
channels of each observation (see ``_feature_observation``/``_grid_observation``
below and ``gym_snake.obs.ego_observation``'s ``opponent_cells`` parameter) so
the policy at least perceives it as "something to avoid", even though it was
never trained with a second snake on the board.

The one exception is ``"rl_trf_battle"``: the same ego/Transformer architecture,
but **fine-tuned in the two-snake battle env** (``examples/train_transformer_battle.py``),
so it was actually trained with a live opponent folded into the very same
``opponent_cells`` channel. It loads and runs through exactly the same code path
as ``"rl_trf"`` (identical observation and action space) -- only the weights
differ.
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
        "file": _EXAMPLES / "ppo_snake_ego.zip",
        "obs": "ego",
        "grid": None,  # ego observation has a fixed shape -> any board size
        "label": "学習済みAI (ego/CNN)",
        # gym_snake supplies both the saved model's feature-extractor class and
        # the shared ego-observation builder.
        "needs_gym_snake": True,
    },
    "rl_trf": {
        "file": _EXAMPLES / "ppo_snake_transformer.zip",
        "obs": "ego",
        "grid": None,
        "label": "学習済みAI (ego/Transformer)",
        "needs_gym_snake": True,  # EgoTransformer class lives in gym_snake
    },
    "rl_trf_battle": {
        # Same EgoTransformer architecture as ``rl_trf`` but fine-tuned in the
        # two-snake battle env (see examples/train_transformer_battle.py), so it
        # actually learned to contest the shared food, dodge the moving
        # opponent and avoid head-on crashes rather than treating the rival as a
        # static wall. Drop-in: identical ego observation and action space, so
        # it runs on any board size and in solo mode too (opponent_cells empty).
        "file": _EXAMPLES / "ppo_snake_transformer_battle.zip",
        "obs": "ego",
        "grid": None,
        "label": "学習済みAI (ego/Transformer 対戦特化)",
        "needs_gym_snake": True,
    },
    "rl_trf_aggr": {
        # Same base EgoTransformer, but a LoRA adapter behavior-cloned from an
        # *aggressive* expert (contests the shared food and crowds the
        # opponent), then merged in -- see examples/train_transformer_style.py.
        "file": _EXAMPLES / "ppo_snake_transformer_aggr.zip",
        "obs": "ego",
        "grid": None,
        "label": "学習済みAI (Transformer 攻撃型)",
        "needs_gym_snake": True,
    },
    "rl_trf_def": {
        # Same base EgoTransformer with a LoRA adapter cloned from a *defensive*
        # expert (yields contested food, keeps distance and survives longer).
        "file": _EXAMPLES / "ppo_snake_transformer_def.zip",
        "obs": "ego",
        "grid": None,
        "label": "学習済みAI (Transformer 防御型)",
        "needs_gym_snake": True,
    },
    "rl_trf_bal": {
        # Neutral middle style: a LoRA adapter cloned from the *balanced* expert
        # (the base search AI's own ranking), so the three playstyle models
        # share one base and differ only by the demonstrator's personality.
        "file": _EXAMPLES / "ppo_snake_transformer_bal.zip",
        "obs": "ego",
        "grid": None,
        "label": "学習済みAI (Transformer バランス型)",
        "needs_gym_snake": True,
    },
}

# Public set of RL strategy names for dispatchers (ai.choose, views).
MODEL_NAMES = frozenset(_MODELS)


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


def _opponent_cells(state: GameState, snake_index: int) -> list[tuple[int, int]]:
    """Cells occupied by every *other* living snake, treated as a static
    hazard (see the module docstring for why this is the extent of RL
    opponent-awareness)."""
    return [
        cell
        for j, s in enumerate(state.snakes)
        if j != snake_index and s.alive
        for cell in s.body
    ]


def build_observation(state: GameState, obs_type: str, snake_index: int):
    """Convert a Django game state into the requested gym_snake observation
    for ``state.snakes[snake_index]``."""
    if obs_type == "grid":
        return _grid_observation(state, snake_index)
    if obs_type == "ego":
        # Single source of truth: the exact function SnakeEnv uses in training.
        from gym_snake.obs import ego_observation

        me = state.snakes[snake_index]
        return ego_observation(
            me.body, state.food, state.grid, _heading_index(me.direction),
            opponent_cells=_opponent_cells(state, snake_index),
        )
    return _feature_observation(state, snake_index)


def _feature_observation(state: GameState, snake_index: int):
    """Recreate gym_snake's 11-feature observation from a Django game state.

    Byte-for-byte consistent with ``SnakeEnv._feature_obs`` for a single
    snake; with more than one snake, the other snake's body cells are folded
    into the same "danger" sensors as this snake's own body.
    """
    import numpy as np

    me = state.snakes[snake_index]
    head = me.body[0]
    hidx = _heading_index(me.direction)
    heading = _HEADINGS[hidx]
    right = _HEADINGS[(hidx + 1) % 4]
    left = _HEADINGS[(hidx - 1) % 4]
    body = set(me.body[:-1])  # tail cell frees up next tick
    body.update(_opponent_cells(state, snake_index))
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
        float(food is not None and food[0] < head[0]),  # food is left
        float(food is not None and food[0] > head[0]),  # food is right
        float(food is not None and food[1] < head[1]),  # food is up
        float(food is not None and food[1] > head[1]),  # food is down
    ]
    return np.asarray(feats, dtype=np.float32)


def _grid_observation(state: GameState, snake_index: int):
    """Recreate gym_snake's (3, H, W) grid observation from a Django state.

    Byte-for-byte consistent with ``SnakeEnv._grid_obs`` for a single snake;
    channel 0 = body, channel 1 = head, channel 2 = food. With more than one
    snake, the other snake's body cells are added to the body channel too.
    """
    import numpy as np

    me = state.snakes[snake_index]
    g = state.grid
    obs = np.zeros((3, g, g), dtype=np.float32)
    for (x, y) in me.body:
        obs[0, y, x] = 1.0
    for (x, y) in _opponent_cells(state, snake_index):
        obs[0, y, x] = 1.0
    hx, hy = me.body[0]
    obs[1, hy, hx] = 1.0
    if state.food is not None:
        fx, fy = state.food
        obs[2, fy, fx] = 1.0
    return obs


# -- action selection ------------------------------------------------------


def choose_direction(state: GameState, snake_index: int, name: str = "rl") -> str:
    """Return the absolute direction model ``name`` would take for
    ``state.snakes[snake_index]``."""
    model = _load_model(name)
    obs = build_observation(state, _MODELS[name]["obs"], snake_index)
    action, _ = model.predict(obs, deterministic=True)

    me = state.snakes[snake_index]
    hidx = _heading_index(me.direction)
    a = int(action)
    if a == 1:            # turn right (clockwise)
        hidx = (hidx + 1) % 4
    elif a == 2:          # turn left (counter-clockwise)
        hidx = (hidx - 1) % 4
    return _DIR_NAMES[hidx]
