"""Tests for the two-snake battle environment (SnakeBattleEnv).

These need only gymnasium + numpy + the ``game`` engine (no torch / SB3), so
they run wherever the base test suite runs. The self-play ``model:`` opponent
path is exercised by the training script's own smoke, not here.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_snake  # noqa: F401  (registers the env)
from gym_snake.envs import SnakeBattleEnv
from gym_snake.envs.battle_env import make_opponent


def test_registered_make():
    env = gym.make("gym_snake/SnakeBattle-v0", grid_size=10)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (5, 11, 11)  # same fixed ego shape as the single-snake env
    env.close()


def test_check_env_search_opponent():
    check_env(SnakeBattleEnv(grid_size=8, opponent="search"), skip_render_check=True)


def test_check_env_random_opponent():
    check_env(SnakeBattleEnv(grid_size=8, opponent="random"), skip_render_check=True)


def test_reset_spawns_two_snakes():
    env = SnakeBattleEnv(grid_size=12)
    env.reset(seed=0)
    assert len(env._state.snakes) == 2
    assert all(s.alive for s in env._state.snakes)


def test_reset_is_deterministic_with_seed():
    a = SnakeBattleEnv(grid_size=10)
    b = SnakeBattleEnv(grid_size=10)
    obs_a, _ = a.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_step_returns_five_tuple():
    env = SnakeBattleEnv(grid_size=8)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {"score", "opp_score", "alive", "win"} <= set(info)


def test_opponent_body_appears_in_observation():
    """The live opponent must be folded into the deadly channel (channel 0),
    otherwise the policy is blind to it -- the whole point of battle training."""
    env = SnakeBattleEnv(grid_size=9)
    env.reset(seed=3)
    agent, opp = env._state.snakes
    # Put the opponent directly in front of the agent's head so it lands inside
    # the local window and must show up as deadly.
    hx, hy = agent.body[0]
    opp.body = [(hx + 1, hy), (hx + 2, hy), (hx + 3, hy)]
    obs = env._agent_obs()
    # Deadly channel (0) must contain more than just the agent's own body/walls.
    with_opp = obs[0].sum()
    opp.body = []
    without_opp = env._agent_obs()[0].sum()
    assert with_opp > without_opp


def test_eating_food_rewards():
    env = SnakeBattleEnv(grid_size=9, reward_shaping=False)
    env.reset(seed=0)
    agent = env._state.snakes[0]
    hx, hy = agent.body[0]
    # Food straight ahead (agent 0 spawns heading right).
    env._state.food = (hx + 1, hy)
    _, reward, _, _, info = env.step(0)  # straight
    assert reward == env.REWARD_FOOD + env.REWARD_STEP
    assert info["score"] == 1


def test_wall_collision_terminates_with_penalty():
    env = SnakeBattleEnv(grid_size=7, opponent="search")
    env.reset(seed=0)
    terminated = False
    reward = 0.0
    for _ in range(env.grid_size + 3):
        _, reward, terminated, _, _ = env.step(0)  # straight into the wall
        if terminated:
            break
    assert terminated
    assert reward <= env.REWARD_DEATH + env.REWARD_STEP + 1e-6


def test_win_lose_bonus_applied_at_terminal():
    """A terminal win pays reward_win on top of the base reward."""
    env = SnakeBattleEnv(grid_size=6, opponent="search",
                         reward_win=5.0, reward_lose=-5.0, reward_shaping=False)
    env.reset(seed=0)
    # Force a state where the agent is ahead, then drive it into the wall.
    env._state.snakes[0].score = 30
    env._state.snakes[1].score = 0
    reward = 0.0
    for _ in range(env.grid_size + 3):
        _, reward, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    # Death penalty (-1) plus the +5 win bonus -> net positive.
    assert reward > 0


def test_opponent_specs_build():
    for spec in ("search", "random"):
        fn = make_opponent(spec)
        assert callable(fn)
    try:
        make_opponent("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ansi_render():
    env = SnakeBattleEnv(grid_size=6, render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "@" in out and "#" in out  # both heads


def test_agent_survives_when_opponent_dies():
    """The episode must NOT end just because the opponent died -- the agent
    keeps playing solo, mirroring the Django battle rules."""
    env = SnakeBattleEnv(grid_size=10, opponent="search")
    env.reset(seed=0)
    env._state.snakes[1].alive = False
    env._state.snakes[1].body = []
    _, _, terminated, truncated, info = env.step(0)
    assert info["opp_alive"] is False
    assert not (terminated or truncated)  # agent is still alive and playing


def test_opponent_channels_change_the_observation_space():
    plain = SnakeBattleEnv(grid_size=12)
    aware = SnakeBattleEnv(grid_size=12, opponent_channels=True)
    assert plain.observation_space.shape == (5, 11, 11)
    assert aware.observation_space.shape == (8, 11, 11)
    check_env(aware, skip_render_check=True)


def test_opponent_channels_locate_the_rival_head():
    """The rival's head gets its own channel instead of being merged into ours."""
    env = SnakeBattleEnv(grid_size=12, opponent_channels=True)
    obs, _ = env.reset(seed=3)
    state = env._state
    assert obs.shape == (8, 11, 11)
    # Both snakes are alive at reset, so the rival is somewhere on the radar.
    assert state.snakes[1].alive
    assert obs[7].max() == 1.0            # minimap always carries it
    assert obs[6].sum() in (0.0, 1.0)     # local head: at most one pixel
    # And the dead-opponent case leaves them empty rather than lying.
    state.snakes[1].alive = False
    state.snakes[1].body = []
    assert env._agent_obs()[5:].sum() == 0.0
