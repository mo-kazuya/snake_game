"""Server-side AI that drives the snake.

Strategy (in priority order):

1. **Seek food safely.** Find the shortest path to the food with BFS. Only
   take it if, after eating, the snake can still reach its own tail — i.e. it
   won't seal itself into a pocket. This "can I still find my tail?" check is
   the classic trick that keeps a greedy snake alive.

2. **Survive.** If there is no safe path to the food, move toward the tail /
   into the largest open area (measured by flood fill) to buy time until a
   safe path opens up.

3. **Last resort.** If nothing is safe, take any legal move so the engine —
   not the AI — decides the game is over.
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


def _simulate(snake, direction, food, grid):
    """Return the snake body *after* moving one step, or None if illegal."""
    dx, dy = DIRECTIONS[direction]
    head = (snake[0][0] + dx, snake[0][1] + dy)
    if not (0 <= head[0] < grid and 0 <= head[1] < grid):
        return None
    ate = head == food
    body = list(snake) if ate else list(snake[:-1])
    if head in body:
        return None
    return [head] + (list(snake) if ate else list(snake[:-1]))


def _reachable_tail(snake, food, grid):
    """After a hypothetical move, can the head still reach the tail cell?

    If yes, the snake still has room to follow its tail out of any pocket, so
    the move is considered safe.
    """
    head, tail = snake[0], snake[-1]
    # The tail will move out of the way next tick, so treat it as free.
    blocked = set(snake[:-1])
    return _bfs(head, tail, blocked, grid) is not None


def _opposite(direction, current):
    dx, dy = DIRECTIONS[direction]
    cx, cy = DIRECTIONS[current]
    return (dx, dy) == (-cx, -cy)


def choose(state: GameState, strategy: str = "search") -> tuple[str, str]:
    """Pick a direction using the requested strategy.

    ``strategy`` is either ``"search"`` (the BFS/flood-fill AI below) or
    ``"rl"`` (the trained PPO policy in :mod:`game.rl_agent`). If ``"rl"`` is
    requested but unavailable — or it errors at runtime — this transparently
    falls back to the search AI.

    Returns ``(direction, strategy_used)`` so the caller can tell whether a
    fallback happened.
    """
    if strategy == "rl":
        from . import rl_agent

        if rl_agent.is_available():
            try:
                return rl_agent.choose_direction(state), "rl"
            except Exception:
                # Any inference failure should never break the game loop.
                pass
    return choose_direction(state), "search"


def choose_direction(state: GameState) -> str:
    """Pick the next direction for the snake given the current game state."""
    grid = state.grid
    snake = state.snake
    head = snake[0]
    food = state.food

    legal = [
        d
        for d in DIRECTIONS
        if not _opposite(d, state.direction) and _simulate(snake, d, food, grid)
    ]
    if not legal:
        # Boxed in — return current heading and let the engine end the game.
        return state.direction

    # 1. Shortest safe path to the food.
    body_blocked = set(snake[:-1])  # tail cell is free next tick
    path = _bfs(head, food, body_blocked, grid)
    if path:
        first = path[0]
        move = _cell_to_direction(head, first)
        if move in legal:
            after = _simulate(snake, move, food, grid)
            if after and _reachable_tail(after, food, grid):
                return move

    # 2. Survival: follow the move that keeps the most open space.
    def openness(direction: str) -> int:
        after = _simulate(snake, direction, food, grid)
        if after is None:
            return -1
        blocked = set(after[:-1])
        return _flood_fill_size(after[0], blocked, grid)

    # Prefer moves that keep the tail reachable, then maximize open space.
    def score(direction: str) -> tuple[int, int]:
        after = _simulate(snake, direction, food, grid)
        tail_ok = 1 if (after and _reachable_tail(after, food, grid)) else 0
        return (tail_ok, openness(direction))

    return max(legal, key=score)


def _cell_to_direction(frm, to):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    for name, vec in DIRECTIONS.items():
        if vec == (dx, dy):
            return name
    return None
