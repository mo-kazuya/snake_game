"""In-memory store of active games.

Games are kept in a process-local dict keyed by a UUID. This is intentionally
simple: it is perfect for the single-process development server and small
demos. For a multi-process / multi-worker deployment you would swap this for a
shared backend (Redis, the database, Django's cache framework, etc.) behind the
same tiny interface.
"""

from __future__ import annotations

import threading
import uuid

from .engine import GameState

_lock = threading.Lock()
_games: dict[str, GameState] = {}

# Cap the number of concurrent games so a long-running server can't grow the
# dict without bound. Oldest games are evicted first.
_MAX_GAMES = 500


def create(grid: int = 20) -> tuple[str, GameState]:
    game_id = uuid.uuid4().hex
    state = GameState(grid=grid)
    with _lock:
        if len(_games) >= _MAX_GAMES:
            oldest = next(iter(_games))
            _games.pop(oldest, None)
        _games[game_id] = state
    return game_id, state


def get(game_id: str) -> GameState | None:
    with _lock:
        return _games.get(game_id)


def remove(game_id: str) -> None:
    with _lock:
        _games.pop(game_id, None)
