"""Tests for the egocentric observation (gym_snake.obs)."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_snake.envs import SnakeEnv
from gym_snake.obs import CHANNELS, WINDOW, ego_observation

C = WINDOW // 2  # head pixel (row, col) = (C, C)

# Heading vectors in SnakeEnv order: 0=up, 1=right, 2=down, 3=left.
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def test_shape_is_constant_across_board_sizes():
    for grid in (8, 10, 20, 30):
        snake = [(grid // 2, grid // 2), (grid // 2 - 1, grid // 2)]
        obs = ego_observation(snake, (0, 0), grid, heading_idx=1)
        assert obs.shape == (CHANNELS, WINDOW, WINDOW)
        assert obs.dtype == np.float32


def test_front_right_left_alignment_for_all_headings():
    """A deadly cell ahead/right/left of the head must land on fixed pixels."""
    grid = 15
    head = (7, 7)
    for hidx, (dx, dy) in enumerate(HEADINGS):
        right = HEADINGS[(hidx + 1) % 4]
        left = HEADINGS[(hidx - 1) % 4]
        for vec, pixel in (
            ((dx, dy), (C - 1, C)),          # straight ahead -> up in image
            (right, (C, C + 1)),             # right-hand side -> right in image
            (left, (C, C - 1)),              # left-hand side -> left in image
        ):
            blocker = (head[0] + vec[0], head[1] + vec[1])
            # snake: head + a far-away tail so blocker is a free-standing body
            # cell of "another" part (we place it via a longer body list).
            snake = [head, blocker, (0, 0)]  # blocker is snake[1] (not tail)
            obs = ego_observation(snake, None, grid, hidx)
            assert obs[0][pixel] == 1.0, (hidx, vec, pixel)


def test_walls_appear_ahead():
    """Heading right, head on the right edge: the wall ahead must be deadly."""
    grid = 9
    snake = [(8, 4), (7, 4), (6, 4)]
    obs = ego_observation(snake, None, grid, heading_idx=1)
    assert obs[0][C - 1, C] == 1.0  # cell ahead is out of bounds -> wall


def test_food_ahead_lands_up_in_image():
    grid = 15
    head = (7, 7)
    for hidx, (dx, dy) in enumerate(HEADINGS):
        food = (head[0] + 2 * dx, head[1] + 2 * dy)  # two cells ahead
        snake = [head, (head[0] - dx, head[1] - dy)]
        obs = ego_observation(snake, food, grid, hidx)
        assert obs[1][C - 2, C] == 1.0, hidx


def test_tail_cell_is_not_deadly():
    grid = 9
    snake = [(4, 4), (3, 4), (2, 4)]  # tail at (2,4)
    obs = ego_observation(snake, None, grid, heading_idx=1)
    # Tail is 2 cells behind the head: in the image that's 2 below center.
    assert obs[0][C + 2, C] == 0.0
    # The neck (1 behind) IS deadly.
    assert obs[0][C + 1, C] == 1.0


def test_minimap_sees_far_food():
    """Food outside the local window must still show up on the minimap."""
    grid = 30
    snake = [(2, 2), (1, 2)]
    food = (28, 28)  # far outside the 11x11 local view
    obs = ego_observation(snake, food, grid, heading_idx=1)
    assert obs[1].max() == 0.0   # not in local view
    assert obs[3].max() == 1.0   # visible on the minimap (peak-normalized)


def test_no_food_leaves_food_channels_empty():
    snake = [(4, 4), (3, 4)]
    obs = ego_observation(snake, None, 9, heading_idx=1)
    assert obs[1].max() == 0.0 and obs[3].max() == 0.0


def test_env_checker_ego():
    check_env(SnakeEnv(grid_size=8, obs_type="ego"), skip_render_check=True)
    check_env(SnakeEnv(grid_size=20, obs_type="ego"), skip_render_check=True)


def test_env_ego_obs_matches_helper():
    env = SnakeEnv(grid_size=12, obs_type="ego")
    env.reset(seed=3)
    expected = ego_observation(env.snake, env.food, env.grid_size, env.heading_idx)
    assert np.array_equal(env._get_obs(), expected)
