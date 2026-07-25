"""A two-snake ("battle") Gymnasium environment for self-play / opponent-aware
training.

The shipped RL policies (features/MLP, ego/CNN, ego/Transformer) were all
trained **single-snake** in :class:`~gym_snake.envs.snake_env.SnakeEnv`. In the
Django battle mode the opponent is only folded into the existing "danger"
channels as a static hazard (see ``game.rl_agent`` and
``gym_snake.obs.ego_observation``'s ``opponent_cells``), so those policies never
actually *learned* to contest the shared food, dodge a moving opponent, or avoid
a head-on collision.

This environment closes that gap. It is a thin Gymnasium wrapper over the
**authoritative two-snake engine** (:class:`game.engine.GameState`), so the
training-time rules are byte-for-byte the same as what the Django server runs:

* one shared food item — whoever reaches it first eats it,
* death from a wall, any snake's body (own or the opponent's), or a head-on
  collision (both heads onto the same cell — both die),
* a snake keeps playing while the other is dead; the game ends when both are.

The **agent controls snake 0**; snake 1 is driven by a configurable *opponent
policy* (the BFS search AI by default, a random-safe walker, or a frozen PPO
snapshot for self-play — see :func:`make_opponent`). Because the observation is
the exact same fixed-shape ``ego_observation`` (now populated with the live
opponent body via ``opponent_cells``), a model trained here is **drop-in
compatible** with the Django ego adapter and can be **warm-started from the
existing single-snake Transformer** — same observation space, same action space.

Actions are the same relative ``Discrete(3)`` (``0=straight, 1=right,
2=left``) as :class:`SnakeEnv`, and the base reward matches ``SnakeEnv`` exactly
so a warm-started value head stays calibrated; optional battle-specific bonuses
(opponent death / win / loss) are off by default and enabled by the trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# The engine and search AI live in the Django app (``game`` package) at the repo
# root. gym_snake is normally pip-installed *editable* from that same repo, so
# derive the root from this file and make ``game`` importable even when the env
# is constructed inside a fresh SubprocVecEnv worker.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gym_snake.obs import CHANNELS, check_window, ego_observation  # noqa: E402

# Absolute headings as (dx, dy), clockwise -- identical to SnakeEnv._HEADINGS.
#   index: 0=up  1=right  2=down  3=left
_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_DIR_NAMES = ["up", "right", "down", "left"]


def _heading_index(direction: str) -> int:
    from game.engine import DIRECTIONS

    return _HEADINGS.index(DIRECTIONS[direction])


# ---------------------------------------------------------------------------
# Opponent policies
# ---------------------------------------------------------------------------

def make_opponent(spec: str):
    """Build the opponent's move function from a small string ``spec``.

    ``spec`` is one of:

    * ``"search"``    — the BFS/flood-fill AI (``game.ai.choose_direction``),
      the same strong opponent the Django server uses. **Default.**
    * ``"random"``    — a uniformly random *safe* move (never an immediate
      wall/body suicide when a safe move exists), for curriculum variety.
    * ``"model:<path>"`` — a frozen PPO checkpoint loaded on the CPU and queried
      with the ego observation, for **self-play**. Loaded lazily on first use so
      the spec stays picklable across ``SubprocVecEnv`` workers and no torch
      import happens unless self-play is actually requested.

    Returns a callable ``fn(state, idx) -> direction_name`` for snake ``idx``.
    """
    if spec == "search":
        def _search(state, idx):
            from game.ai import choose_direction

            return choose_direction(state, idx)

        return _search

    if spec == "random":
        import random

        def _random(state, idx):
            from game.ai import _opposite, _simulate
            from game.engine import DIRECTIONS

            me = state.snakes[idx]
            other = {
                cell
                for j, s in enumerate(state.snakes)
                if j != idx and s.alive
                for cell in s.body
            }
            safe = [
                d for d in DIRECTIONS
                if not _opposite(d, me.direction)
                and _simulate(me.body, d, state.food, state.grid, other)
            ]
            if safe:
                return random.choice(safe)
            # Boxed in: any non-reversing move, else current heading.
            legal = [d for d in DIRECTIONS if not _opposite(d, me.direction)]
            return random.choice(legal) if legal else me.direction

        return _random

    if spec.startswith("model:"):
        path = spec[len("model:"):]
        cache: dict = {}

        def _model(state, idx):
            model = cache.get("m")
            if model is None:
                import gym_snake.policies  # noqa: F401  (registers extractors)
                from stable_baselines3 import PPO

                model = PPO.load(path, device="cpu")
                cache["m"] = model
            me = state.snakes[idx]
            opp = [
                cell
                for j, s in enumerate(state.snakes)
                if j != idx and s.alive
                for cell in s.body
            ]
            obs = ego_observation(
                me.body, state.food, state.grid,
                _heading_index(me.direction), opponent_cells=opp,
            )
            # Frozen opponent plays stochastically for exploration diversity.
            action, _ = model.predict(obs, deterministic=False)
            hidx = _heading_index(me.direction)
            a = int(action)
            if a == 1:
                hidx = (hidx + 1) % 4
            elif a == 2:
                hidx = (hidx - 1) % 4
            return _DIR_NAMES[hidx]

        return _model

    raise ValueError(f"unknown opponent spec: {spec!r}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SnakeBattleEnv(gym.Env):
    """Two-snake Snake with the agent controlling snake 0.

    Parameters
    ----------
    grid_size:
        Side length of the square board.
    opponent:
        Opponent spec string for :func:`make_opponent` (``"search"`` by
        default), or an already-built ``fn(state, idx) -> direction`` callable.
    ego_window:
        Side of the ego observation (odd, >= 5); ``None`` uses
        :data:`gym_snake.obs.WINDOW`. Must match the window the policy being
        trained / warm-started was built for.
    reward_shaping:
        Add SnakeEnv's small toward/away-from-food nudge each step.
    max_steps_without_food:
        Stall truncation for the agent; defaults to ``grid_size**2`` (the same
        backstop the engine enforces).
    reward_opp_death, reward_win, reward_lose:
        Optional battle-specific bonuses (all ``0.0`` by default so the base
        reward is identical to :class:`SnakeEnv`). ``reward_opp_death`` is paid
        once, the tick the opponent dies while the agent is alive; ``reward_win``
        / ``reward_lose`` are paid at episode end by comparing final scores.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    # Base reward constants -- identical to SnakeEnv so warm-starting a
    # single-snake value head stays calibrated.
    REWARD_FOOD = 1.0
    REWARD_DEATH = -1.0
    REWARD_STEP = -0.005
    REWARD_SHAPING = 0.05

    def __init__(
        self,
        grid_size: int = 12,
        opponent: str | object = "search",
        ego_window: int | None = None,
        reward_shaping: bool = True,
        max_steps_without_food: int | None = None,
        reward_opp_death: float = 0.0,
        reward_win: float = 0.0,
        reward_lose: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 5:
            raise ValueError("grid_size must be >= 5")

        self.grid_size = grid_size
        self.ego_window = check_window(ego_window)
        self._opponent_spec = opponent
        self._opponent_fn = opponent if callable(opponent) else None
        self.reward_shaping = reward_shaping
        self.max_steps_without_food = (
            max_steps_without_food
            if max_steps_without_food is not None
            else grid_size * grid_size
        )
        self.reward_opp_death = reward_opp_death
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(CHANNELS, self.ego_window, self.ego_window), dtype=np.float32,
        )

        self._state = None            # game.engine.GameState
        self._steps_since_food = 0
        self._opp_was_alive = True

    # -- opponent (lazy build so string specs survive pickling) ------------

    def _opponent(self, state, idx):
        if self._opponent_fn is None:
            self._opponent_fn = make_opponent(self._opponent_spec)
        return self._opponent_fn(state, idx)

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        from game.engine import GameState

        # Seed the engine's module-level RNG (food placement / spawn) so
        # episodes are reproducible when a seed is given.
        if seed is not None:
            import random as _random

            _random.seed(seed)

        self._state = GameState(grid=self.grid_size)
        self._state.reset(["agent", "opponent"])
        self._steps_since_food = 0
        self._opp_was_alive = self._state.snakes[1].alive

        obs = self._agent_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action: {action!r}")

        state = self._state
        agent, opp = state.snakes[0], state.snakes[1]

        head_before = agent.body[0]
        prev_dist = self._food_distance(head_before)

        # Agent's relative action -> absolute direction from its heading.
        directions = [self._rel_to_dir(agent.direction, int(action)), opp.direction]
        # Opponent's move (only if it is still alive).
        if opp.alive:
            try:
                directions[1] = self._opponent(state, 1)
            except Exception:
                # An opponent failure must never break training; hold heading.
                directions[1] = opp.direction

        events = state.step(directions)
        ev = events[0]

        reward = self.REWARD_STEP
        terminated = False
        truncated = False

        if ev["ate"]:
            reward += self.REWARD_FOOD
            self._steps_since_food = 0
        elif ev["dead"]:
            if agent.stalled:
                # Stall backstop -> treat as truncation, no death penalty
                # (matches SnakeEnv's truncation-without-penalty semantics).
                truncated = True
            else:
                reward += self.REWARD_DEATH
                terminated = True
        else:
            self._steps_since_food += 1
            if self.reward_shaping and state.food is not None:
                new_dist = self._food_distance(agent.body[0])
                reward += (
                    self.REWARD_SHAPING if new_dist < prev_dist else -self.REWARD_SHAPING
                )

        # Battle bonus: opponent just died while the agent is alive.
        if (
            self.reward_opp_death
            and self._opp_was_alive
            and not state.snakes[1].alive
            and agent.alive
        ):
            reward += self.reward_opp_death
        self._opp_was_alive = state.snakes[1].alive

        # Stall truncation (independent of the engine's own backstop, so it
        # fires even if the engine constant were ever changed).
        if not (terminated or truncated) and self._steps_since_food >= self.max_steps_without_food:
            truncated = True

        # The whole game being over (both dead, or board filled) ends the
        # episode too.
        if state.game_over and not (terminated or truncated):
            terminated = True

        # Terminal win/lose bonus based on final scores.
        if terminated or truncated:
            if self.reward_win or self.reward_lose:
                if agent.score > opp.score:
                    reward += self.reward_win
                elif agent.score < opp.score:
                    reward += self.reward_lose

        obs = self._agent_obs()
        info = self._get_info()
        info["death"] = None if ev["moved"] or not ev["dead"] else (
            "stall" if agent.stalled else "collision"
        )
        return obs, reward, terminated, truncated, info

    # -- observations & helpers -------------------------------------------

    def _agent_obs(self) -> np.ndarray:
        state = self._state
        me = state.snakes[0]
        if not me.body:  # dead: body is cleared by the engine
            return np.zeros(
                (CHANNELS, self.ego_window, self.ego_window), dtype=np.float32
            )
        opp_cells = [
            cell for cell in state.snakes[1].body if state.snakes[1].alive
        ]
        return ego_observation(
            me.body, state.food, state.grid,
            _heading_index(me.direction), window=self.ego_window,
            opponent_cells=opp_cells,
        )

    def _rel_to_dir(self, direction: str, action: int) -> str:
        hidx = _heading_index(direction)
        if action == 1:
            hidx = (hidx + 1) % 4
        elif action == 2:
            hidx = (hidx - 1) % 4
        return _DIR_NAMES[hidx]

    def _food_distance(self, cell) -> int:
        food = self._state.food
        if food is None:
            return 0
        return abs(cell[0] - food[0]) + abs(cell[1] - food[1])

    def _get_info(self) -> dict:
        state = self._state
        agent, opp = state.snakes[0], state.snakes[1]
        # Engine score is +10 per food; report "food eaten" for readability and
        # keep the raw scores for the win/lose gate.
        win = None
        if state.game_over or not agent.alive:
            win = agent.score > opp.score
        return {
            "score": agent.score // 10,
            "opp_score": opp.score // 10,
            "length": len(agent.body),
            "alive": agent.alive,
            "opp_alive": opp.alive,
            "steps": state.steps,
            "win": win,
        }

    # -- rendering ---------------------------------------------------------

    def render(self):
        if self.render_mode != "ansi":
            return None
        state = self._state
        g = state.grid
        heads = {s.body[0]: i for i, s in enumerate(state.snakes) if s.body}
        bodies = {
            cell: i
            for i, s in enumerate(state.snakes)
            for cell in s.body[1:]
        }
        rows = []
        for y in range(g):
            row = []
            for x in range(g):
                if (x, y) in heads:
                    row.append("@" if heads[(x, y)] == 0 else "#")
                elif (x, y) in bodies:
                    row.append("o" if bodies[(x, y)] == 0 else "x")
                elif state.food is not None and (x, y) == state.food:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        info = self._get_info()
        return "\n".join(rows) + f"\nyou={info['score']} opp={info['opp_score']}"
