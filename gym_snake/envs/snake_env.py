"""A Gymnasium environment for the game of Snake.

This is a self-contained, dependency-light environment suitable for training
reinforcement-learning agents. It follows the modern Gymnasium API:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

Design choices that make it RL-friendly
----------------------------------------
* **Relative actions** ``Discrete(3)``: ``0 = 直進 (straight)``, ``1 = 右折
  (turn right)``, ``2 = 左折 (turn left)``. Because the agent can never choose
  to reverse into itself, there are no "instant death" illegal moves to learn
  around.
* **Two observation modes** (``obs_type``):
    - ``"grid"`` (default): a ``(3, H, W)`` float32 image with channels
      ``[body, head, food]`` — good for CNN policies.
    - ``"features"``: an 11-dim vector (danger sensors + heading one-hot + food
      direction) — good for fast MLP policies.
* **Shaped reward** to speed up learning: a large bonus for eating, a penalty
  for dying, and (optionally) a small nudge toward the food each step.
* **Truncation** when the snake wanders too long without eating, so episodes
  during training always terminate.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Absolute headings as (dx, dy), ordered clockwise so relative turns are easy.
# index:      0=up       1=right    2=down     3=left
_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeEnv(gym.Env):
    """Snake as a Gymnasium environment.

    Parameters
    ----------
    grid_size:
        Side length of the square board (number of cells). Default 12.
    obs_type:
        ``"grid"`` (3xHxW image) or ``"features"`` (11-dim vector).
    reward_shaping:
        If True, add a small +/- reward for moving closer to / farther from the
        food. Helps early learning; turn off for a "pure" reward.
    max_steps_without_food:
        Episode is truncated if this many steps pass without eating. Defaults to
        ``grid_size * grid_size``.
    render_mode:
        One of ``"ansi"``, ``"rgb_array"``, ``"human"`` or ``None``.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    # Reward constants.
    REWARD_FOOD = 1.0
    REWARD_DEATH = -1.0
    REWARD_STEP = -0.005          # small time pressure
    REWARD_SHAPING = 0.05         # magnitude of the toward/away-from-food nudge

    def __init__(
        self,
        grid_size: int = 12,
        obs_type: str = "grid",
        reward_shaping: bool = True,
        max_steps_without_food: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 5:
            raise ValueError("grid_size must be >= 5")
        if obs_type not in ("grid", "features", "ego"):
            raise ValueError("obs_type must be 'grid', 'features' or 'ego'")

        self.grid_size = grid_size
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.max_steps_without_food = (
            max_steps_without_food
            if max_steps_without_food is not None
            else grid_size * grid_size
        )
        self.render_mode = render_mode

        # Action space: relative turn.
        self.action_space = spaces.Discrete(3)

        # Observation space depends on the chosen encoding.
        if obs_type == "grid":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(3, grid_size, grid_size), dtype=np.float32,
            )
        elif obs_type == "ego":
            # Egocentric view + minimap: fixed shape for EVERY board size, so
            # one trained model transfers across grid sizes (see gym_snake.obs).
            from gym_snake import obs as ego_obs

            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(ego_obs.CHANNELS, ego_obs.WINDOW, ego_obs.WINDOW),
                dtype=np.float32,
            )
        else:  # "features"
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(11,), dtype=np.float32,
            )

        # State (populated in reset()).
        self.snake: list[tuple[int, int]] = []
        self.heading_idx: int = 1  # start heading right
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self._steps_since_food: int = 0
        self._window = None  # lazy pygame window for human render

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.heading_idx = 1  # right
        self.score = 0
        self.steps = 0
        self._steps_since_food = 0
        self._place_food()

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action: {action!r}")

        # Apply relative turn: 0 straight, 1 right (clockwise), 2 left (ccw).
        if action == 1:
            self.heading_idx = (self.heading_idx + 1) % 4
        elif action == 2:
            self.heading_idx = (self.heading_idx - 1) % 4

        dx, dy = _HEADINGS[self.heading_idx]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        prev_dist = self._food_distance(head)

        terminated = False
        reward = self.REWARD_STEP
        self.steps += 1
        self._steps_since_food += 1

        # Wall collision.
        out_of_bounds = not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        )
        ate = (not out_of_bounds) and new_head == self.food
        # Body collision (the tail moves away unless we grow).
        body = self.snake if ate else self.snake[:-1]
        hit_self = (not out_of_bounds) and new_head in body

        if out_of_bounds or hit_self:
            terminated = True
            reward += self.REWARD_DEATH
            obs = self._get_obs()
            info = self._get_info()
            info["death"] = "wall" if out_of_bounds else "self"
            if self.render_mode == "human":
                self.render()
            return obs, reward, terminated, False, info

        # Advance the snake.
        self.snake.insert(0, new_head)
        if ate:
            self.score += 1
            self._steps_since_food = 0
            reward += self.REWARD_FOOD
            self._place_food()
            if self.food is None:  # board full -> the agent solved it
                terminated = True
        else:
            self.snake.pop()
            if self.reward_shaping:
                new_dist = self._food_distance(new_head)
                reward += self.REWARD_SHAPING if new_dist < prev_dist else -self.REWARD_SHAPING

        truncated = self._steps_since_food >= self.max_steps_without_food

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # -- observations & info ----------------------------------------------

    def _get_obs(self):
        if self.obs_type == "grid":
            return self._grid_obs()
        if self.obs_type == "ego":
            from gym_snake.obs import ego_observation

            return ego_observation(
                self.snake, self.food, self.grid_size, self.heading_idx
            )
        return self._feature_obs()

    def _grid_obs(self) -> np.ndarray:
        g = self.grid_size
        obs = np.zeros((3, g, g), dtype=np.float32)
        for (x, y) in self.snake:
            obs[0, y, x] = 1.0            # body channel
        hx, hy = self.snake[0]
        obs[1, hy, hx] = 1.0              # head channel
        if self.food is not None:
            fx, fy = self.food
            obs[2, fy, fx] = 1.0          # food channel
        return obs

    def _feature_obs(self) -> np.ndarray:
        """Classic 11-feature encoding used in many Snake RL tutorials."""
        head = self.snake[0]
        heading = _HEADINGS[self.heading_idx]
        right = _HEADINGS[(self.heading_idx + 1) % 4]
        left = _HEADINGS[(self.heading_idx - 1) % 4]

        def danger(vec):
            nx, ny = head[0] + vec[0], head[1] + vec[1]
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                return 1.0
            return 1.0 if (nx, ny) in self.snake[:-1] else 0.0

        food = self.food if self.food is not None else head
        feats = [
            danger(heading),                              # danger straight
            danger(right),                                # danger right
            danger(left),                                 # danger left
            float(heading == _HEADINGS[0]),               # heading up
            float(heading == _HEADINGS[1]),               # heading right
            float(heading == _HEADINGS[2]),               # heading down
            float(heading == _HEADINGS[3]),               # heading left
            float(food[0] < head[0]),                     # food is left
            float(food[0] > head[0]),                     # food is right
            float(food[1] < head[1]),                     # food is up
            float(food[1] > head[1]),                     # food is down
        ]
        return np.asarray(feats, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "length": len(self.snake),
            "steps": self.steps,
            "head": self.snake[0],
            "food": self.food,
        }

    # -- helpers -----------------------------------------------------------

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        if not free:
            self.food = None
            return
        idx = self.np_random.integers(len(free))
        self.food = free[int(idx)]

    def _food_distance(self, cell) -> int:
        if self.food is None:
            return 0
        return abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1])

    # -- rendering ---------------------------------------------------------

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def _render_ansi(self) -> str:
        g = self.grid_size
        head = self.snake[0]
        body = set(self.snake[1:])
        rows = []
        for y in range(g):
            row = []
            for x in range(g):
                if (x, y) == head:
                    row.append("@")
                elif (x, y) in body:
                    row.append("o")
                elif self.food is not None and (x, y) == self.food:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        border = "+" + "-" * g + "+"
        body_str = "\n".join("|" + r + "|" for r in rows)
        return f"{border}\n{body_str}\n{border}\nscore={self.score} len={len(self.snake)}"

    def _render_rgb(self) -> np.ndarray:
        cell = 16
        g = self.grid_size
        img = np.zeros((g * cell, g * cell, 3), dtype=np.uint8)
        img[:] = (15, 52, 96)  # board background

        def fill(x, y, color):
            img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = color

        if self.food is not None:
            fill(self.food[0], self.food[1], (244, 63, 94))
        for i, (x, y) in enumerate(self.snake):
            fill(x, y, (74, 222, 128) if i == 0 else (34, 197, 94))
        return img

    def _render_human(self):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise gym.error.DependencyNotInstalled(
                "human rendering needs pygame: pip install pygame"
            ) from exc
        frame = self._render_rgb()
        h, w, _ = frame.shape
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("gym_snake")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surf, (0, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(self.metadata["render_fps"])

    def close(self):
        if self._window is not None:  # pragma: no cover
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
