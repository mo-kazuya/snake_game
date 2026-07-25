"""A two-snake ("duel") Gymnasium environment for fine-tuning.

The single-snake :class:`gym_snake.envs.SnakeEnv` can't teach a policy to
compete with another snake. This environment reuses the *exact* authoritative
two-snake rules from the Django engine (:class:`game.engine.GameState`) so the
training dynamics match the deployed game, and it presents the learner the
*same* egocentric observation the Django adapter serves at play time -- with
the opponent's body folded into the deadly/body channels via
``gym_snake.obs.ego_observation(opponent_cells=...)``.

Layout
------
* The **agent** always controls ``snakes[0]`` and sees the ego observation for
  it. Actions are the usual relative ``Discrete(3)`` (straight / right / left).
* The **opponent** controls ``snakes[1]``:
    - ``opponent="search"``  -> the BFS/flood-fill AI (``game.ai``), a strong,
      opponent-aware baseline;
    - ``opponent="model"``   -> a frozen PPO policy loaded from
      ``opponent_model_path`` (used for self-play; refreshable at runtime via
      :meth:`refresh_opponent` so a VecEnv can swap in a newer snapshot).

The episode (from the agent's point of view) ends when ``snakes[0]`` dies --
wall, any body, head-on collision, or the per-snake stall timeout -- or when
the board is cleared. If the opponent dies first, the agent simply plays on
solo for the rest of the episode, which is exactly what happens in the real
game.

Reward mirrors ``SnakeEnv`` (food +1, death -1, small step cost, optional
toward-food shaping); the competitive pressure comes for free from the shared
food and the fact that colliding with the opponent is fatal.
"""

from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.engine import DIRECTIONS, GameState
from gym_snake.obs import CHANNELS, WINDOW, ego_observation

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_DIR_NAMES = ["up", "right", "down", "left"]

REWARD_FOOD = 1.0
REWARD_DEATH = -1.0
REWARD_STEP = -0.005
REWARD_SHAPING = 0.05


def _heading_index(direction: str) -> int:
    return _HEADINGS.index(DIRECTIONS[direction])


class SnakeDuelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 20,
        opponent: str = "search",
        opponent_model_path: str | None = None,
        reward_shaping: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.opponent_kind = opponent          # "search" or "model"
        self.opponent_model_path = opponent_model_path
        self.reward_shaping = reward_shaping

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(CHANNELS, WINDOW, WINDOW), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._opponent_model = None
        self.state: GameState | None = None
        if seed is not None:
            random.seed(seed)

    # -- opponent management (self-play) -----------------------------------

    def _load_opponent(self) -> None:
        from stable_baselines3 import PPO

        import gym_snake.policies  # noqa: F401  (registers the extractor class)

        self._opponent_model = PPO.load(self.opponent_model_path, device="cpu")

    def refresh_opponent(self, path: str) -> None:
        """Swap in a newer self-play snapshot (called via VecEnv.env_method).

        No-op for envs whose opponent is the search AI, so a VecEnv can mix
        search-opponent and self-play envs and broadcast refreshes to all.
        """
        if self.opponent_kind != "model":
            return
        self.opponent_model_path = path
        self._load_opponent()

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.state = GameState(grid=self.grid_size)
        # Strategy labels are irrelevant to the engine's rules; the AIs here
        # are driven externally by this env, not by state.snakes[i].strategy.
        self.state.reset(["agent", "opponent"])
        if self.opponent_kind == "model" and self._opponent_model is None:
            self._load_opponent()
        return self._obs(0), self._info()

    def step(self, action):
        s = self.state
        me = s.snakes[0]
        food = s.food
        prev_dist = self._dist(me.body[0], food)

        my_dir = self._rel_to_dir(0, int(action))
        opp_dir = self._opponent_dir()
        events = s.step([my_dir, opp_dir])
        ev = events[0]

        reward = REWARD_STEP
        terminated = False
        if ev["ate"]:
            reward += REWARD_FOOD
        elif ev["dead"]:
            reward += REWARD_DEATH
            terminated = True
        elif self.reward_shaping and food is not None:
            new_dist = self._dist(s.snakes[0].body[0], food)
            reward += REWARD_SHAPING if new_dist < prev_dist else -REWARD_SHAPING

        if s.game_over:          # board cleared, or both snakes dead
            terminated = True

        obs = self._obs(0) if s.snakes[0].alive else self._empty_obs()
        return obs, reward, terminated, False, self._info()

    # -- helpers ------------------------------------------------------------

    def _empty_obs(self):
        return np.zeros((CHANNELS, WINDOW, WINDOW), dtype=np.float32)

    def _obs(self, i: int):
        s = self.state
        me = s.snakes[i]
        opponent = [
            cell
            for j, o in enumerate(s.snakes)
            if j != i and o.alive
            for cell in o.body
        ]
        return ego_observation(
            me.body, s.food, s.grid, _heading_index(me.direction),
            opponent_cells=opponent,
        )

    def _rel_to_dir(self, i: int, action: int) -> str:
        hidx = _heading_index(self.state.snakes[i].direction)
        if action == 1:
            hidx = (hidx + 1) % 4
        elif action == 2:
            hidx = (hidx - 1) % 4
        return _DIR_NAMES[hidx]

    def _opponent_dir(self) -> str:
        s = self.state
        opp = s.snakes[1]
        if not opp.alive:
            return opp.direction  # ignored by the engine for a dead snake
        if self.opponent_kind == "model" and self._opponent_model is not None:
            # Stochastic sampling gives the learner a less exploitable, more
            # varied opponent than greedy argmax.
            action, _ = self._opponent_model.predict(self._obs(1), deterministic=False)
            return self._rel_to_dir(1, int(action))
        from game.ai import choose_direction
        return choose_direction(s, 1)

    @staticmethod
    def _dist(cell, food) -> int:
        if food is None:
            return 0
        return abs(cell[0] - food[0]) + abs(cell[1] - food[1])

    def _info(self) -> dict:
        s = self.state
        return {
            "score": s.snakes[0].score,
            "opp_score": s.snakes[1].score,
            "opp_alive": s.snakes[1].alive,
            "length": len(s.snakes[0].body),
        }
