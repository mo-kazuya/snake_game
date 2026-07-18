"""Server-side AI that drives a snake.

Each snake picks its move independently, given the *whole* game state, so it
treats the other living snake's body as just another obstacle (in addition
to walls and its own body). Strategy (in priority order):

1. **Seek food safely.** Find the shortest path to the food with BFS. Only
   take it if, after eating, the snake can still reach its own tail — i.e. it
   won't seal itself into a pocket. This "can I still find my tail?" check is
   the classic trick that keeps a greedy snake alive.

2. **Survive.** If there is no safe path to the food, move toward the tail /
   into the largest open area (measured by flood fill) to buy time until a
   safe path opens up. Ties are broken in favor of cells not recently
   visited, so a snake with no goal-directed path doesn't settle into
   repeating the exact same loop forever (GameState's stall timeout in
   engine.py is the hard backstop for pockets with no way out at all).

3. **Last resort.** If nothing is safe, take any legal move so the engine —
   not the AI — decides the game is over.

The trained RL policies (:mod:`game.rl_agent`) were trained single-snake and
can't be retrained here, so they only get a lighter form of opponent
awareness: the other snake's body is folded into their existing "danger" /
"body" observation channels (see rl_agent.py and gym_snake/obs.py) rather
than a real architecture change.
"""

from __future__ import annotations

from collections import deque

from .engine import DIRECTIONS, GameState


def _neighbors(cell, grid):
    x, y = cell
    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid and 0 <= ny < grid:
            yield (nx, ny)


def _bfs(start, goal, blocked, grid):
    """Shortest path from ``start`` to ``goal`` avoiding ``blocked`` cells.

    Returns the list of cells from start (exclusive) to goal (inclusive), or
    ``None`` if unreachable.
    """
    if start == goal:
        return []
    queue = deque([start])
    came_from = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in _neighbors(cur, grid):
            if nxt in came_from or (nxt in blocked and nxt != goal):
                continue
            came_from[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while came_from[path[-1]] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _flood_fill_size(start, blocked, grid):
    """Count cells reachable from ``start`` without entering ``blocked``."""
    if start in blocked:
        return 0
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in _neighbors(cur, grid):
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _simulate(snake, direction, food, grid, extra_blocked=frozenset()):
    """Return the snake body *after* moving one step, or None if illegal.

    ``extra_blocked`` is additional static obstacles -- in practice the other
    snake's current body -- that also make the move illegal.
    """
    dx, dy = DIRECTIONS[direction]
    head = (snake[0][0] + dx, snake[0][1] + dy)
    if not (0 <= head[0] < grid and 0 <= head[1] < grid):
        return None
    if head in extra_blocked:
        return None
    ate = head == food
    body = list(snake) if ate else list(snake[:-1])
    if head in body:
        return None
    return [head] + (list(snake) if ate else list(snake[:-1]))


def _reachable_tail(snake, food, grid, extra_blocked=frozenset()):
    """After a hypothetical move, can the head still reach the tail cell?

    If yes, the snake still has room to follow its tail out of any pocket, so
    the move is considered safe.
    """
    head, tail = snake[0], snake[-1]
    # The tail will move out of the way next tick, so treat it as free.
    blocked = set(snake[:-1]) | extra_blocked
    return _bfs(head, tail, blocked, grid) is not None


def _opposite(direction, current):
    dx, dy = DIRECTIONS[direction]
    cx, cy = DIRECTIONS[current]
    return (dx, dy) == (-cx, -cy)


def _record_head(state: GameState, snake_index: int) -> deque:
    """Per-snake short-term memory of recently visited head cells.

    Stashed directly on the ``GameState`` instance rather than a module-level
    registry, so its lifetime is simply tied to the game's own -- no cleanup
    needed when a game is evicted from the store. Pure AI bookkeeping, not
    authoritative game state, so it deliberately isn't part of
    ``to_dict()``/``from_dict()``.
    """
    memory = getattr(state, "_ai_recent_heads", None)
    if memory is None:
        memory = {}
        state._ai_recent_heads = memory
    recent = memory.get(snake_index)
    if recent is None:
        recent = deque(maxlen=max(32, state.grid * 2))
        memory[snake_index] = recent
    recent.append(state.snakes[snake_index].body[0])
    return recent


def _other_bodies(state: GameState, snake_index: int) -> set[tuple[int, int]]:
    """All cells occupied by every *other* living snake (treated as static
    obstacles -- there's no model of what the opponent will do this tick, so
    the conservative choice is to avoid its whole current body, tail
    included)."""
    return {
        cell
        for j, s in enumerate(state.snakes)
        if j != snake_index and s.alive
        for cell in s.body
    }


def choose(state: GameState, snake_index: int, strategy: str = "search") -> tuple[str, str]:
    """Pick a direction for ``state.snakes[snake_index]`` using ``strategy``.

    ``strategy`` is ``"search"`` (the BFS/flood-fill AI below) or one of the
    trained PPO policies in :mod:`game.rl_agent` (``"rl"`` = features/MLP,
    ``"rl_cnn"`` = ego/CNN, ``"rl_trf"`` = ego/Transformer). If an RL strategy
    is requested but unavailable, runs on the wrong board size, or errors at
    runtime, this transparently falls back to the search AI.

    Returns ``(direction, strategy_used)`` so the caller can tell whether a
    fallback happened.
    """
    from . import rl_agent

    if strategy in rl_agent.MODEL_NAMES:
        req = rl_agent.required_grid(strategy)
        size_ok = req is None or state.grid == req
        if size_ok and rl_agent.is_available(strategy):
            try:
                return rl_agent.choose_direction(state, snake_index, strategy), strategy
            except Exception:
                # Any inference failure should never break the game loop.
                pass
    return choose_direction(state, snake_index), "search"


def choose_direction(state: GameState, snake_index: int = 0) -> str:
    """Pick the next direction for ``state.snakes[snake_index]``."""
    grid = state.grid
    me = state.snakes[snake_index]
    snake = me.body
    head = snake[0]
    food = state.food
    recent = _record_head(state, snake_index)
    other_blocked = _other_bodies(state, snake_index)

    legal = [
        d
        for d in DIRECTIONS
        if not _opposite(d, me.direction)
        and _simulate(snake, d, food, grid, other_blocked)
    ]
    if not legal:
        # Boxed in — return current heading and let the engine end the game.
        return me.direction

    # 1. Shortest safe path to the food.
    body_blocked = set(snake[:-1]) | other_blocked  # tail cell is free next tick
    path = _bfs(head, food, body_blocked, grid)
    if path:
        first = path[0]
        move = _cell_to_direction(head, first)
        if move in legal:
            after = _simulate(snake, move, food, grid, other_blocked)
            if after and _reachable_tail(after, food, grid, other_blocked):
                return move

    # 2. Survival: follow the move that keeps the most open space.
    def openness(direction: str) -> int:
        after = _simulate(snake, direction, food, grid, other_blocked)
        if after is None:
            return -1
        # Exclude the new head itself (`after[0]`) from `blocked`: flood
        # fill starts there, and `_flood_fill_size` short-circuits to 0 if
        # the start cell is in `blocked` -- silently disabling this
        # heuristic for every direction. The tail (`after[-1]`) is excluded
        # too, same "frees up next tick" convention as `_reachable_tail`.
        blocked = set(after[1:-1]) | other_blocked
        return _flood_fill_size(after[0], blocked, grid)

    # Prefer moves that keep the tail reachable, then maximize open space,
    # then prefer a cell not recently visited. Without that last tiebreaker,
    # revisiting the exact same board state would always yield the exact
    # same decision, so a snake with no goal-directed path could repeat the
    # same loop forever.
    def score(direction: str) -> tuple[int, int, int]:
        after = _simulate(snake, direction, food, grid, other_blocked)
        tail_ok = 1 if (after and _reachable_tail(after, food, grid, other_blocked)) else 0
        fresh = 0 if (after and after[0] in recent) else 1
        return (tail_ok, openness(direction), fresh)

    return max(legal, key=score)


def _cell_to_direction(frm, to):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    for name, vec in DIRECTIONS.items():
        if vec == (dx, dy):
            return name
    return None
