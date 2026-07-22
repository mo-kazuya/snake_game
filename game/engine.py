"""Snake game engine.

Holds the authoritative game state and the rules for advancing it one step.
The engine is deliberately UI-agnostic: it only knows about a grid, one or
more snakes, food, and scores. The AI (see ``ai.py``) decides *which*
direction each snake moves; the engine applies those moves simultaneously
and reports what happened.

Two snakes share one board and one food item -- whichever reaches it first
eats it. A snake dies from a wall collision, running into any snake's body
(its own or the other's), or a head-on collision with the other snake's new
head. A dead snake's body is removed from the board so it stops blocking the
survivor. The game ends once every snake is dead.
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
class Snake:
    """One snake's state within a (possibly multi-snake) game."""

    body: list[tuple[int, int]] = field(default_factory=list)
    direction: str = "right"
    strategy: str = "search"
    score: int = 0
    steps_since_food: int = 0
    alive: bool = True
    # True if this snake died from its own stall timeout rather than a
    # collision -- see GameState.step() for why this backstop exists.
    stalled: bool = False

    def _is_opposite(self, direction: str) -> bool:
        dx, dy = DIRECTIONS[direction]
        cx, cy = DIRECTIONS[self.direction]
        return (dx, dy) == (-cx, -cy)

    def to_dict(self) -> dict:
        return {
            "snake": [list(cell) for cell in self.body],
            "direction": self.direction,
            "strategy": self.strategy,
            "score": self.score,
            "steps_since_food": self.steps_since_food,
            "length": len(self.body),
            "alive": self.alive,
            "stalled": self.stalled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Snake":
        return cls(
            body=[tuple(cell) for cell in data["snake"]],
            direction=data["direction"],
            strategy=data.get("strategy", "search"),
            score=data["score"],
            steps_since_food=data.get("steps_since_food", 0),
            alive=data.get("alive", True),
            stalled=data.get("stalled", False),
        )


@dataclass
class GameState:
    """Authoritative state for a game of Snake, shared by every snake in it."""

    grid: int = 20
    snakes: list[Snake] = field(default_factory=list)
    food: tuple[int, int] | None = (0, 0)
    steps: int = 0
    game_over: bool = False

    def __post_init__(self) -> None:
        if not self.snakes:
            self.reset(["search", "search"])

    # -- lifecycle ---------------------------------------------------------

    def reset(self, strategies: list[str]) -> None:
        """Start a new game with one snake per entry in ``strategies``.

        A single strategy spawns one snake dead-center, same as the original
        single-snake layout. Two or more spawn on separate rows, facing away
        from each other, so they don't start overlapping or immediately
        head-to-head.
        """
        mid = self.grid // 2
        solo = len(strategies) == 1
        self.snakes = []
        for i, strategy in enumerate(strategies):
            row = mid if solo else (mid - 2 if i % 2 == 0 else mid + 2)
            heading = "right" if i % 2 == 0 else "left"
            dx = -1 if heading == "right" else 1
            body = [(mid + dx * j, row) for j in range(3)]
            self.snakes.append(
                Snake(body=body, direction=heading, strategy=strategy)
            )
        self.steps = 0
        self.game_over = False
        self.place_food()

    def place_food(self) -> None:
        """Put food on a random cell free of every snake (or end the game if full)."""
        occupied = {cell for s in self.snakes for cell in s.body}
        free = [
            (x, y)
            for x in range(self.grid)
            for y in range(self.grid)
            if (x, y) not in occupied
        ]
        if not free:
            # Board is completely filled: the snakes have "won".
            self.food = None
            self.game_over = True
            return
        self.food = random.choice(free)

    # -- stepping ------------------------------------------------------------

    def step(self, directions: list[str]) -> list[dict]:
        """Advance one tick. ``directions`` has one entry per snake (dead or
        already-finished snakes ignore theirs). Returns one event dict per
        snake, in the same order as ``self.snakes``."""
        if self.game_over:
            return [{"moved": False, "ate": False, "dead": True} for _ in self.snakes]

        living = [i for i, s in enumerate(self.snakes) if s.alive]

        # 1. Turn (ignoring 180-degree reversals) and compute candidate heads.
        new_heads: dict[int, tuple[int, int]] = {}
        for i in living:
            s = self.snakes[i]
            if not s._is_opposite(directions[i]):
                s.direction = directions[i]
            dx, dy = DIRECTIONS[s.direction]
            hx, hy = s.body[0]
            new_heads[i] = (hx + dx, hy + dy)

        dead_this_tick: set[int] = set()

        # 2. Wall collisions.
        for i, head in new_heads.items():
            if not (0 <= head[0] < self.grid and 0 <= head[1] < self.grid):
                dead_this_tick.add(i)

        # 3. Head-on collisions: two survivors moving onto the same cell
        # crash into each other and both die (no one eats there either).
        for i in living:
            if i in dead_this_tick:
                continue
            for j in living:
                if j <= i or j in dead_this_tick:
                    continue
                if new_heads[i] == new_heads[j]:
                    dead_this_tick.add(i)
                    dead_this_tick.add(j)

        # 4. Food: whichever surviving candidate's head lands on it (at most
        # one, since a shared landing cell was just resolved as a head-on
        # collision above).
        eaten_by = None
        for i in living:
            if i not in dead_this_tick and new_heads[i] == self.food:
                eaten_by = i
                break

        # 5. Body collisions: a snake dies if its new head lands on any
        # snake's body (its own or another's), tail excluded unless that
        # snake is growing this tick (tail is about to vacate otherwise).
        blocked_by: dict[int, set[tuple[int, int]]] = {}
        for i in living:
            if i in dead_this_tick:
                continue
            s = self.snakes[i]
            blocked_by[i] = set(s.body if i == eaten_by else s.body[:-1])

        for i in living:
            if i in dead_this_tick:
                continue
            head = new_heads[i]
            if any(head in blocked for blocked in blocked_by.values()):
                dead_this_tick.add(i)

        # 6. Apply moves for everyone still alive.
        events = [{"moved": False, "ate": False, "dead": True} for _ in self.snakes]
        for i in living:
            s = self.snakes[i]
            if i in dead_this_tick:
                s.alive = False
                s.body = []
                events[i] = {"moved": False, "ate": False, "dead": True}
                continue

            ate = i == eaten_by
            s.body.insert(0, new_heads[i])
            if ate:
                s.score += 10
                s.steps_since_food = 0
            else:
                s.body.pop()
                s.steps_since_food += 1
                if s.steps_since_food >= self.grid * self.grid:
                    s.alive = False
                    s.stalled = True
                    s.body = []
                    events[i] = {"moved": False, "ate": False, "dead": True}
                    continue

            events[i] = {"moved": True, "ate": ate, "dead": False}

        self.steps += 1
        if eaten_by is not None:
            self.place_food()  # may itself set game_over if the board is now full

        if all(not s.alive for s in self.snakes):
            self.game_over = True

        return events

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "grid": self.grid,
            "snakes": [s.to_dict() for s in self.snakes],
            "food": list(self.food) if self.food is not None else None,
            "steps": self.steps,
            "game_over": self.game_over,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        state = cls(grid=data["grid"], snakes=[Snake.from_dict(d) for d in data["snakes"]])
        state.food = tuple(data["food"]) if data.get("food") is not None else None
        state.steps = data.get("steps", 0)
        state.game_over = data["game_over"]
        return state
