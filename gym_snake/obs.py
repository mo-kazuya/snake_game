"""Egocentric ("ego") observation for Snake.

This encodes the board relative to the snake itself, which makes a CNN policy
both **easy to train** and **board-size independent by construction**:

* a fixed ``WINDOW x WINDOW`` **local view** centered on the head and rotated so
  the snake always "looks up" — the cell straight ahead is always the same
  pixel, whatever the heading or board size;
* a fixed ``WINDOW x WINDOW`` **minimap** of the whole board (area-downscaled or
  nearest-upscaled), rotated into the same frame, so the policy can see where
  the food is even when it is outside the local view.

Because the output shape ``(5, WINDOW, WINDOW)`` never depends on the board
size, one trained model runs on 10x10 and 20x20 boards alike — no adaptive
pooling tricks needed, and a plain flatten CNN head works.

``WINDOW`` is only the *default*: :func:`ego_observation` takes a ``window``
argument, and every consumer (``SnakeEnv``/``SnakeBattleEnv`` via ``ego_window``,
the Django adapter via the per-model ``window`` entry in ``game.rl_agent``)
threads it through. A larger odd window widens the local view (more cells of
real geometry around the head) *and* refines the minimap (the board is
downscaled less), at a quadratic cost in tokens for the Transformer policy. A
model must always be fed the same window it was trained with.

Channels:

====  =========================================================
 0    local: deadly cells (walls + snake body, tail excluded)
 1    local: food
 2    minimap: snake body (including tail)
 3    minimap: food (peak-normalized)
 4    minimap: head (peak-normalized)
====  =========================================================

The module is pure numpy so the Django server can build the exact same
observation without importing gymnasium/torch. Both ``SnakeEnv`` and the Django
adapter call :func:`ego_observation` — a single source of truth.

Heading indices follow ``SnakeEnv``: ``0=up, 1=right, 2=down, 3=left``.
"""

from __future__ import annotations

import numpy as np

WINDOW = 11  # side of both the local view and the minimap (odd: head-centered)
CHANNELS = 5


def check_window(window: int | None) -> int:
    """Validate an ego window size, returning it (``None`` -> :data:`WINDOW`).

    The window must be **odd** (so the head sits on the exact center pixel and
    "straight ahead" is always the same pixel) and at least 5.
    """
    if window is None:
        return WINDOW
    window = int(window)
    if window < 5 or window % 2 == 0:
        raise ValueError(f"ego window must be an odd int >= 5, got {window}")
    return window


def _resize(arr: np.ndarray, out: int) -> np.ndarray:
    """Resize a square ``(H, H)`` array to ``(out, out)``.

    Downscaling uses exact area averaging; upscaling (H < out) uses
    nearest-neighbor sampling. Pure numpy, no interpolation deps.
    """
    h = arr.shape[0]
    if h == out:
        return arr.astype(np.float32)
    if h < out:
        idx = (np.arange(out) * h) // out
        return arr[np.ix_(idx, idx)].astype(np.float32)
    edges = np.linspace(0, h, out + 1).astype(int)
    starts = edges[:-1]
    sums = np.add.reduceat(np.add.reduceat(arr, starts, axis=0), starts, axis=1)
    counts = np.diff(edges)
    area = np.outer(counts, counts)
    return (sums / area).astype(np.float32)


def _peak_normalize(arr: np.ndarray) -> np.ndarray:
    peak = arr.max()
    return arr / peak if peak > 0 else arr


def ego_observation(
    snake: list[tuple[int, int]],
    food: tuple[int, int] | None,
    grid_size: int,
    heading_idx: int,
    window: int = WINDOW,
    opponent_cells: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the ``(5, window, window)`` egocentric observation.

    ``snake`` is head-first ``(x, y)`` cells, ``heading_idx`` uses the SnakeEnv
    convention (0=up, 1=right, 2=down, 3=left). ``food`` may be ``None`` (full
    board), leaving the food channels empty.

    ``opponent_cells``, if given, marks extra cells (in practice another
    snake's body, in a multi-snake game) as deadly/body in exactly the same
    way as this snake's own body. ``None`` (the default) reproduces the
    original single-snake observation exactly, so training (``SnakeEnv``)
    and any existing single-snake caller are unaffected.
    """
    c = window // 2

    deadly = np.zeros((grid_size, grid_size), dtype=np.float32)
    body_full = np.zeros_like(deadly)
    food_map = np.zeros_like(deadly)
    head_map = np.zeros_like(deadly)

    for (x, y) in snake[:-1]:  # the tail cell frees up next tick
        deadly[y, x] = 1.0
    for (x, y) in snake:
        body_full[y, x] = 1.0
    for (x, y) in (opponent_cells or ()):
        deadly[y, x] = 1.0
        body_full[y, x] = 1.0
    hx, hy = snake[0]
    head_map[hy, hx] = 1.0
    if food is not None:
        food_map[food[1], food[0]] = 1.0

    # Local view: pad the deadly map with walls (=1), food with empties (=0),
    # then cut the window around the head.
    deadly_pad = np.pad(deadly, c, constant_values=1.0)
    food_pad = np.pad(food_map, c, constant_values=0.0)
    ly, lx = hy + c, hx + c  # head position in padded coordinates
    local_deadly = deadly_pad[ly - c: ly + c + 1, lx - c: lx + c + 1]
    local_food = food_pad[ly - c: ly + c + 1, lx - c: lx + c + 1]

    # Rotate everything into the "heading up" frame. With np.rot90 (counter-
    # clockwise) k = heading_idx puts the cell in front of the head at
    # (c-1, c), the right-hand cell at (c, c+1), the left-hand at (c, c-1).
    k = heading_idx
    obs = np.stack(
        [
            np.rot90(local_deadly, k),
            np.rot90(local_food, k),
            np.rot90(_resize(body_full, window), k),
            np.rot90(_peak_normalize(_resize(food_map, window)), k),
            np.rot90(_peak_normalize(_resize(head_map, window)), k),
        ]
    )
    return np.ascontiguousarray(obs, dtype=np.float32)
