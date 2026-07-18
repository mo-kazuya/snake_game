"""Snake game engine.

Holds the authoritative game state and the rules for advancing it one step.
The engine is deliberately UI-agnostic: it only knows about a grid, a snake,
food, and a score. The AI (see ``ai.py``) decides *which* direction to move;
the engine applies that move and reports what happened.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Direction vectors keyed by a short name.
DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class GameState:
    """Authoritative state for a single game of Snake."""

    grid: int = 20
    snake: list[tuple[int, int]] = field(default_factory=list)
    direction: str = "right"
    food: tuple[int, int] = (0, 0)
    score: int = 0
    steps: int = 0
    game_over: bool = False

    def __post_init__(self) -> None:
        if not self.snake:
            self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        mid = self.grid // 2
        # Start length 3, heading right, comfortably inside the board.
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction = "right"
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.place_food()

    def place_food(self) -> None:
        """Put food on a random empty cell (or mark a win if the board is full)."""
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid)
            for y in range(self.grid)
            if (x, y) not in occupied
        ]
        if not free:
            # Board is completely filled: the snake has "won".
            self.game_over = True
            return
        self.food = random.choice(free)

    # -- stepping ----------------------------------------------------------

    def step(self, direction: str) -> dict:
        """Advance one tick in ``direction``. Returns a small event dict."""
        if self.game_over:
            return {"moved": False, "ate": False, "dead": True}

        # Ignore a 180-degree reversal; keep the current heading instead.
        if not self._is_opposite(direction):
            self.direction = direction

        dx, dy = DIRECTIONS[self.direction]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        # Wall collision.
        if not (0 <= new_head[0] < self.grid and 0 <= new_head[1] < self.grid):
            self.game_over = True
            return {"moved": False, "ate": False, "dead": True}

        ate = new_head == self.food
        # Body collision. The tail cell is free *unless* we grow this tick.
        body = self.snake if ate else self.snake[:-1]
        if new_head in body:
            self.game_over = True
            return {"moved": False, "ate": False, "dead": True}

        self.snake.insert(0, new_head)
        self.steps += 1
        if ate:
            self.score += 10
            self.place_food()
        else:
            self.snake.pop()

        return {"moved": True, "ate": ate, "dead": self.game_over}

    def _is_opposite(self, direction: str) -> bool:
        dx, dy = DIRECTIONS[direction]
        cx, cy = DIRECTIONS[self.direction]
        return (dx, dy) == (-cx, -cy)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "grid": self.grid,
            "snake": [list(cell) for cell in self.snake],
            "direction": self.direction,
            "food": list(self.food),
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake),
            "game_over": self.game_over,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        state = cls(grid=data["grid"])
        state.snake = [tuple(cell) for cell in data["snake"]]
        state.direction = data["direction"]
        state.food = tuple(data["food"])
        state.score = data["score"]
        state.steps = data.get("steps", 0)
        state.game_over = data["game_over"]
        return state
