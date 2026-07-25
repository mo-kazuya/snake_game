"""Tests for the Snake engine, AI, and HTTP/WebSocket endpoints."""

from __future__ import annotations

import json
from collections import deque
from unittest import mock

from channels.testing import WebsocketCommunicator
from django.test import TestCase

from snakeai.asgi import application

from . import ai, rl_agent
from .engine import GameState, Snake


class EngineTests(TestCase):
    def test_initial_state(self):
        s = GameState(grid=20)
        self.assertEqual(len(s.snakes), 2)
        for snake in s.snakes:
            self.assertEqual(len(snake.body), 3)
            self.assertTrue(snake.alive)
        self.assertFalse(s.game_over)
        # Food is never placed on either snake, and the snakes don't overlap.
        occupied = {cell for snake in s.snakes for cell in snake.body}
        self.assertEqual(len(occupied), 6)
        self.assertNotIn(s.food, occupied)

    def test_reset_solo_spawns_one_centered_snake(self):
        s = GameState(grid=20)
        s.reset(["search"])
        self.assertEqual(len(s.snakes), 1)
        self.assertEqual(len(s.snakes[0].body), 3)
        self.assertEqual(s.snakes[0].body[0], (10, 10))  # dead center
        self.assertNotIn(s.food, set(s.snakes[0].body))

    def test_reset_battle_spawns_two_non_overlapping_snakes(self):
        s = GameState(grid=20)
        s.reset(["search", "rl_cnn"])
        self.assertEqual(len(s.snakes), 2)
        self.assertEqual(s.snakes[0].strategy, "search")
        self.assertEqual(s.snakes[1].strategy, "rl_cnn")
        occupied = {cell for snake in s.snakes for cell in snake.body}
        self.assertEqual(len(occupied), 6)  # no overlap

    def test_solo_game_ends_when_the_only_snake_dies(self):
        s = GameState(grid=10, snakes=[Snake(body=[(9, 5), (8, 5), (7, 5)], direction="right")])
        s.step(["right"])  # wall collision
        self.assertFalse(s.snakes[0].alive)
        self.assertTrue(s.game_over)

    def test_move_advances_head(self):
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        head = s.snakes[0].body[0]
        s.step(["right"])
        self.assertEqual(s.snakes[0].body[0], (head[0] + 1, head[1]))
        self.assertEqual(len(s.snakes[0].body), 3)  # no growth without food

    def test_wall_collision_kills_snake(self):
        s = GameState(grid=10, snakes=[Snake(body=[(9, 5), (8, 5), (7, 5)], direction="right")])
        events = s.step(["right"])
        self.assertTrue(events[0]["dead"])
        self.assertFalse(s.snakes[0].alive)
        # The only snake died, so the game is over too.
        self.assertTrue(s.game_over)

    def test_reversal_is_ignored(self):
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        s.step(["left"])  # 180-degree turn should be ignored
        self.assertEqual(s.snakes[0].direction, "right")

    def test_eating_grows_and_scores(self):
        s = GameState(grid=20, snakes=[Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right")])
        s.food = (6, 5)
        events = s.step(["right"])
        self.assertTrue(events[0]["ate"])
        self.assertEqual(s.snakes[0].score, 10)
        self.assertEqual(len(s.snakes[0].body), 4)

    def test_serialization_roundtrip(self):
        s = GameState(grid=15)
        s.step(["right", "left"])
        restored = GameState.from_dict(s.to_dict())
        self.assertEqual(restored.food, s.food)
        self.assertEqual(restored.game_over, s.game_over)
        self.assertEqual(len(restored.snakes), len(s.snakes))
        for r, o in zip(restored.snakes, s.snakes):
            self.assertEqual(r.body, o.body)
            self.assertEqual(r.direction, o.direction)
            self.assertEqual(r.strategy, o.strategy)
            self.assertEqual(r.score, o.score)
            self.assertEqual(r.steps_since_food, o.steps_since_food)
            self.assertEqual(r.alive, o.alive)
            self.assertEqual(r.stalled, o.stalled)

    def test_stall_without_food_ends_snake(self):
        """A move that doesn't reach food, once too many have piled up,
        must end the snake even though it's perfectly legal (see ai.py's
        anti-loop tiebreaker for why this backstop exists)."""
        s = GameState(grid=10, snakes=[Snake(body=[(5, 5), (5, 6), (5, 7)], direction="up")])
        s.food = (0, 0)
        s.snakes[0].steps_since_food = s.grid * s.grid - 1  # one step from the limit
        events = s.step(["left"])
        self.assertTrue(events[0]["dead"])
        self.assertTrue(s.snakes[0].stalled)
        self.assertTrue(s.game_over)

    def test_stall_flag_absent_on_normal_death(self):
        s = GameState(grid=10, snakes=[Snake(body=[(9, 5), (8, 5), (7, 5)], direction="right")])
        s.step(["right"])  # wall collision
        self.assertFalse(s.snakes[0].stalled)

    def test_eating_resets_stall_counter(self):
        s = GameState(grid=10, snakes=[Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right")])
        s.food = (6, 5)
        s.snakes[0].steps_since_food = 50
        events = s.step(["right"])
        self.assertTrue(events[0]["ate"])
        self.assertEqual(s.snakes[0].steps_since_food, 0)

    # -- two-snake interactions ---------------------------------------------

    def test_two_snakes_share_food_first_to_arrive_eats(self):
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(15, 15), (16, 15), (17, 15)], direction="left"),
        ])
        s.food = (6, 5)  # only snake 0 reaches it this tick
        events = s.step(["right", "left"])
        self.assertTrue(events[0]["ate"])
        self.assertFalse(events[1]["ate"])
        self.assertEqual(s.snakes[0].score, 10)
        self.assertEqual(s.snakes[1].score, 0)
        self.assertIsNotNone(s.food)
        self.assertNotEqual(s.food, (6, 5))  # a new food was placed elsewhere

    def test_head_on_collision_kills_both(self):
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(7, 5), (8, 5), (9, 5)], direction="left"),
        ])
        s.food = (0, 0)  # irrelevant, both crash into cell (6, 5) instead
        events = s.step(["right", "left"])
        self.assertTrue(events[0]["dead"])
        self.assertTrue(events[1]["dead"])
        self.assertFalse(s.snakes[0].alive)
        self.assertFalse(s.snakes[1].alive)
        self.assertTrue(s.game_over)

    def test_crashing_into_other_snakes_body_dies(self):
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(6, 6), (6, 5), (6, 4)], direction="left"),
        ])
        s.food = (0, 0)
        events = s.step(["right", "left"])
        self.assertTrue(events[0]["dead"])   # ran into snake 1's body at (6, 5)
        self.assertFalse(s.snakes[0].alive)
        self.assertFalse(events[1]["dead"])  # snake 1 moved away safely
        self.assertTrue(s.snakes[1].alive)
        # One snake is still alive, so the game continues.
        self.assertFalse(s.game_over)

    def test_dead_snakes_body_is_cleared(self):
        s = GameState(grid=10, snakes=[
            Snake(body=[(9, 5), (8, 5), (7, 5)], direction="right"),
            Snake(body=[(0, 5), (0, 6), (0, 7)], direction="up"),
        ])
        s.food = (5, 9)  # away from both snakes' paths
        s.step(["right", "up"])  # snake 0 hits the wall; snake 1 moves safely
        self.assertEqual(s.snakes[0].body, [])
        self.assertTrue(s.snakes[1].alive)


class AITests(TestCase):
    def test_never_reverses(self):
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        d = ai.choose_direction(s, 0)
        self.assertNotEqual(d, "left")

    def test_moves_toward_food(self):
        s = GameState(grid=20, snakes=[Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right")])
        s.food = (5, 8)  # directly below
        # Straight to food is right/down; the AI should not pick a wasteful up.
        d = ai.choose_direction(s, 0)
        self.assertIn(d, ("down", "right"))

    def test_avoids_immediate_death(self):
        # Snake hugging the right wall heading right -> must not step into wall.
        s = GameState(grid=8, snakes=[Snake(body=[(7, 3), (6, 3), (5, 3)], direction="right")])
        s.food = (7, 0)
        d = ai.choose_direction(s, 0)
        self.assertNotEqual(d, "right")

    def test_avoids_opponent_body(self):
        """A snake must treat the other living snake's body as an obstacle too."""
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(6, 5), (6, 6), (6, 7)], direction="down"),
        ])
        s.food = (10, 10)
        d = ai.choose_direction(s, 0)
        self.assertNotEqual(d, "right")  # (6, 5) belongs to the other snake

    def test_ai_survives_many_steps(self):
        """A full AI-driven game should eat several foods without dying early."""
        s = GameState(grid=12, snakes=[Snake(body=[(6, 6), (5, 6), (4, 6)], direction="right")])
        for _ in range(400):
            if s.game_over:
                break
            s.step([ai.choose_direction(s, 0)])
        # The safe-seeking AI should reach a decent score before any death.
        self.assertGreaterEqual(s.snakes[0].score, 30)

    def test_survival_tiebreak_avoids_recently_visited_cell(self):
        """Ties in survival mode should be broken by recency, not always the
        same fixed direction -- otherwise the AI can settle into repeating
        the exact same loop forever."""
        grid = 21
        mid = grid // 2
        s = GameState(grid=grid, snakes=[
            Snake(body=[(mid, mid), (mid, mid + 1), (mid, mid + 2)], direction="up")
        ])
        s.food = (0, 0)  # irrelevant here; _bfs is patched out below

        # Force branch 2 (survival) regardless of food placement: with no
        # safe path ever found, left/right/up are an exact tie by symmetry
        # on an open board.
        with mock.patch.object(ai, "_bfs", return_value=None):
            baseline = ai.choose_direction(s, 0)

        landing = {
            "up": (mid, mid - 1),
            "left": (mid - 1, mid),
            "right": (mid + 1, mid),
        }[baseline]
        s._ai_recent_heads = {0: deque([landing] * 10, maxlen=64)}

        with mock.patch.object(ai, "_bfs", return_value=None):
            biased = ai.choose_direction(s, 0)

        self.assertNotEqual(biased, baseline)


class RLAgentTests(TestCase):
    def _mirror(self, env):
        """Copy a gym SnakeEnv's live state into a single-snake Django GameState."""
        return GameState(grid=env.grid_size, snakes=[Snake(
            body=list(env.snake),
            direction=rl_agent._DIR_NAMES[env.heading_idx],
        )], food=env.food)

    def test_feature_observation_shape_and_range(self):
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        obs = rl_agent.build_observation(s, "features", 0)
        self.assertEqual(obs.shape, (11,))
        self.assertTrue((obs >= 0).all() and (obs <= 1).all())

    def test_grid_observation_shape(self):
        s = GameState(grid=10, snakes=[Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right")])
        obs = rl_agent.build_observation(s, "grid", 0)
        self.assertEqual(obs.shape, (3, 10, 10))

    def test_feature_observation_matches_gym_encoding(self):
        try:
            from gym_snake.envs import SnakeEnv
        except Exception:
            self.skipTest("gym_snake not importable")
        import numpy as np

        env = SnakeEnv(grid_size=12, obs_type="features")
        env.reset(seed=0)
        s = self._mirror(env)
        self.assertTrue(
            np.array_equal(rl_agent.build_observation(s, "features", 0), env._feature_obs())
        )

    def test_grid_observation_matches_gym_encoding(self):
        try:
            from gym_snake.envs import SnakeEnv
        except Exception:
            self.skipTest("gym_snake not importable")
        import numpy as np

        env = SnakeEnv(grid_size=10, obs_type="grid")
        env.reset(seed=1)
        s = self._mirror(env)
        self.assertTrue(
            np.array_equal(rl_agent.build_observation(s, "grid", 0), env._grid_obs())
        )

    def test_ego_observation_matches_gym_encoding(self):
        """The CNN's ego observation must equal what the env produced in training."""
        try:
            from gym_snake.envs import SnakeEnv
        except Exception:
            self.skipTest("gym_snake not importable")
        import numpy as np

        for grid in (10, 20):
            env = SnakeEnv(grid_size=grid, obs_type="ego")
            env.reset(seed=2)
            s = self._mirror(env)
            self.assertTrue(
                np.array_equal(rl_agent.build_observation(s, "ego", 0), env._get_obs())
            )

    def test_ego_observation_honours_the_model_window(self):
        """A model trained with a wider ego window must be fed that window."""
        try:
            from gym_snake.envs import SnakeEnv
        except Exception:
            self.skipTest("gym_snake not importable")
        import numpy as np

        for window in (11, 21):
            env = SnakeEnv(grid_size=20, obs_type="ego", ego_window=window)
            env.reset(seed=4)
            s = self._mirror(env)
            obs = rl_agent.build_observation(s, "ego", 0, window)
            self.assertEqual(obs.shape, (5, window, window))
            self.assertTrue(np.array_equal(obs, env._get_obs()))

    def test_wide_window_transformer_is_registered(self):
        # The 21x21-window Transformer is a first-class RL strategy, and its
        # registry entry carries the window its weights expect.
        self.assertIn("rl_trf_w21", rl_agent.MODEL_NAMES)
        self.assertEqual(rl_agent._MODELS["rl_trf_w21"]["window"], 21)
        self.assertIsNone(rl_agent.required_grid("rl_trf_w21"))
        meta = rl_agent.strategies_meta()
        self.assertIn("21x21", meta["rl_trf_w21"]["label"])
        self.assertEqual(
            meta["rl_trf_w21"]["available"], rl_agent.is_available("rl_trf_w21")
        )
        # Every other ego model keeps the 11x11 default.
        for name in ("rl_cnn", "rl_trf", "rl_trf_battle", "rl_trf_aggr",
                     "rl_trf_def", "rl_trf_bal", "rl_trf_battle_aggr",
                     "rl_trf_battle_def", "rl_trf_battle_bal"):
            self.assertIsNone(rl_agent._MODELS[name].get("window"))

    def test_opponent_body_folded_into_feature_danger_sensors(self):
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(6, 5), (6, 6), (6, 7)], direction="down"),
        ])
        s.food = (10, 10)
        obs = rl_agent.build_observation(s, "features", 0)
        self.assertEqual(obs[0], 1.0)  # danger straight: (6, 5) is snake 1's head

    def test_required_grid(self):
        # All RL models are board-size independent.
        self.assertIsNone(rl_agent.required_grid("rl"))
        self.assertIsNone(rl_agent.required_grid("rl_cnn"))
        self.assertIsNone(rl_agent.required_grid("rl_trf"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_battle"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_aggr"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_def"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_bal"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_battle_aggr"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_battle_def"))
        self.assertIsNone(rl_agent.required_grid("rl_trf_battle_bal"))

    def test_style_transformers_are_registered(self):
        # The aggressive / defensive / balanced playstyle models are first-class
        # RL strategies with their own labels and availability flags.
        meta = rl_agent.strategies_meta()
        for name, kw in (("rl_trf_aggr", "攻撃"), ("rl_trf_def", "防御"),
                         ("rl_trf_bal", "バランス")):
            self.assertIn(name, rl_agent.MODEL_NAMES)
            self.assertIn(name, meta)
            self.assertIn(kw, meta[name]["label"])
            self.assertIsNone(meta[name]["grid"])
            self.assertEqual(meta[name]["available"], rl_agent.is_available(name))

    def test_battle_style_transformers_are_registered(self):
        # The same three playstyles grown on the battle-trained base are their
        # own strategies, distinct from the single-snake-based ones.
        meta = rl_agent.strategies_meta()
        for name, kw in (("rl_trf_battle_aggr", "攻撃"),
                         ("rl_trf_battle_def", "防御"),
                         ("rl_trf_battle_bal", "バランス")):
            self.assertIn(name, rl_agent.MODEL_NAMES)
            self.assertIn(kw, meta[name]["label"])
            self.assertIn("対戦", meta[name]["label"])
            self.assertIsNone(meta[name]["grid"])
            self.assertEqual(meta[name]["available"], rl_agent.is_available(name))
        # Each style has its own weights file -- no accidental sharing.
        files = {rl_agent._MODELS[n]["file"]
                 for n in ("rl_trf_battle_aggr", "rl_trf_battle_def",
                           "rl_trf_battle_bal", "rl_trf_battle",
                           "rl_trf_aggr", "rl_trf_def", "rl_trf_bal")}
        self.assertEqual(len(files), 7)

    def test_style_transformers_fall_back_when_unavailable(self):
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        for name in ("rl_trf_aggr", "rl_trf_def", "rl_trf_bal",
                     "rl_trf_battle_aggr", "rl_trf_battle_def",
                     "rl_trf_battle_bal"):
            direction, used = ai.choose(s, 0, strategy=name)
            expected = name if rl_agent.is_available(name) else "search"
            self.assertEqual(used, expected)
            self.assertIn(direction, ("up", "down", "left", "right"))

    def test_battle_transformer_is_registered(self):
        # The battle-tuned Transformer is a first-class RL strategy and shows up
        # in the frontend metadata (available flag tracks the weights file).
        self.assertIn("rl_trf_battle", rl_agent.MODEL_NAMES)
        meta = rl_agent.strategies_meta()
        self.assertIn("rl_trf_battle", meta)
        self.assertIn("対戦", meta["rl_trf_battle"]["label"])
        self.assertIsNone(meta["rl_trf_battle"]["grid"])
        self.assertEqual(
            meta["rl_trf_battle"]["available"],
            rl_agent.is_available("rl_trf_battle"),
        )

    def test_choose_falls_back_to_search_when_unknown(self):
        s = GameState(grid=20)
        direction, used = ai.choose(s, 0, strategy="does-not-exist")
        self.assertEqual(used, "search")
        self.assertIn(direction, ("up", "down", "left", "right"))

    def test_rl_returns_valid_direction_when_available(self):
        if not rl_agent.is_available("rl"):
            self.skipTest("features model / stable-baselines3 not available")
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        direction, used = ai.choose(s, 0, strategy="rl")
        self.assertEqual(used, "rl")
        self.assertNotEqual(direction, "left")  # never reverse (starts right)

    def test_cnn_works_on_any_board_size(self):
        if not rl_agent.is_available("rl_cnn"):
            self.skipTest("ego/CNN model not available")
        for grid in (10, 20):
            s = GameState(grid=grid, snakes=[Snake(body=[(6, 6), (5, 6), (4, 6)], direction="right")])
            direction, used = ai.choose(s, 0, strategy="rl_cnn")
            self.assertEqual(used, "rl_cnn", f"grid={grid}")
            self.assertIn(direction, ("up", "down", "right"))  # never reverse

    def test_transformer_works_on_any_board_size(self):
        if not rl_agent.is_available("rl_trf"):
            self.skipTest("ego/Transformer model not available")
        for grid in (10, 20):
            s = GameState(grid=grid, snakes=[Snake(body=[(6, 6), (5, 6), (4, 6)], direction="right")])
            direction, used = ai.choose(s, 0, strategy="rl_trf")
            self.assertEqual(used, "rl_trf", f"grid={grid}")
            self.assertIn(direction, ("up", "down", "right"))  # never reverse

    def test_battle_transformer_works_on_any_board_size(self):
        if not rl_agent.is_available("rl_trf_battle"):
            self.skipTest("battle-tuned Transformer model not available")
        for grid in (10, 20):
            s = GameState(grid=grid, snakes=[Snake(body=[(6, 6), (5, 6), (4, 6)], direction="right")])
            direction, used = ai.choose(s, 0, strategy="rl_trf_battle")
            self.assertEqual(used, "rl_trf_battle", f"grid={grid}")
            self.assertIn(direction, ("up", "down", "right"))  # never reverse

    def test_battle_transformer_falls_back_when_unavailable(self):
        # Until the battle weights ship, selecting the strategy must degrade to
        # the search AI instead of erroring (same graceful path as any missing
        # model). When the weights are present this simply runs the model.
        s = GameState(grid=20, snakes=[Snake(body=[(10, 10), (9, 10), (8, 10)], direction="right")])
        direction, used = ai.choose(s, 0, strategy="rl_trf_battle")
        expected = "rl_trf_battle" if rl_agent.is_available("rl_trf_battle") else "search"
        self.assertEqual(used, expected)
        self.assertIn(direction, ("up", "down", "left", "right"))

    def test_rl_works_with_a_second_snake_on_board(self):
        """RL strategies must not crash just because a second snake exists,
        even though they can't perceive it the way the search AI does."""
        if not rl_agent.is_available("rl_cnn"):
            self.skipTest("ego/CNN model not available")
        s = GameState(grid=20, snakes=[
            Snake(body=[(5, 5), (4, 5), (3, 5)], direction="right"),
            Snake(body=[(15, 15), (14, 15), (13, 15)], direction="right"),
        ])
        direction, used = ai.choose(s, 0, strategy="rl_cnn")
        self.assertEqual(used, "rl_cnn")
        self.assertIn(direction, ("up", "down", "right"))


class ViewTests(TestCase):
    def test_new_game_returns_state(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"grid": 20}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("game_id", data)
        self.assertEqual(data["state"]["grid"], 20)
        self.assertEqual(len(data["state"]["snakes"]), 2)

    def test_new_game_accepts_various_grid_sizes(self):
        for grid in (10, 14, 30, 40):
            res = self.client.post(
                "/api/new/", data=json.dumps({"grid": grid}),
                content_type="application/json",
            )
            data = res.json()
            self.assertEqual(data["state"]["grid"], grid)
            # Both snakes start mid-board with room to move on every size.
            for snake in data["state"]["snakes"]:
                head = snake["snake"][0]
                self.assertTrue(0 < head[0] < grid and 0 < head[1] < grid)

    def test_new_game_clamps_grid(self):
        for sent, expected in ((2, 8), (100, 50)):
            res = self.client.post(
                "/api/new/", data=json.dumps({"grid": sent}),
                content_type="application/json",
            )
            self.assertEqual(res.json()["state"]["grid"], expected)

    def test_new_game_invalid_grid_falls_back_to_default(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"grid": "abc"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["state"]["grid"], 20)

    def test_new_game_accepts_per_snake_strategies(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"strategies": ["search", "rl_cnn"]}),
            content_type="application/json",
        )
        data = res.json()["state"]
        self.assertEqual(len(data["snakes"]), 2)
        self.assertEqual(data["snakes"][0]["strategy"], "search")
        self.assertEqual(data["snakes"][1]["strategy"], "rl_cnn")

    def test_new_game_solo_spawns_one_snake(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"strategies": ["search"]}),
            content_type="application/json",
        )
        data = res.json()["state"]
        self.assertEqual(len(data["snakes"]), 1)
        self.assertEqual(data["snakes"][0]["strategy"], "search")

    def test_new_game_defaults_to_two_snakes(self):
        res = self.client.post("/api/new/", content_type="application/json")
        self.assertEqual(len(res.json()["state"]["snakes"]), 2)

    def test_new_game_caps_at_two_snakes(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"strategies": ["search", "search", "search"]}),
            content_type="application/json",
        )
        self.assertEqual(len(res.json()["state"]["snakes"]), 2)

    def test_new_game_ignores_malformed_strategies(self):
        res = self.client.post(
            "/api/new/", data=json.dumps({"strategies": "not-a-list"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()["state"]
        self.assertEqual(len(data["snakes"]), 2)
        self.assertEqual(data["snakes"][0]["strategy"], "search")

    def test_index_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "AIスネークゲーム")


class GameConsumerTests(TestCase):
    """The per-step move now travels over a WebSocket (game/consumers.py)."""

    async def test_step_advances_game(self):
        new = self.client.post("/api/new/", content_type="application/json").json()
        gid = new["game_id"]
        communicator = WebsocketCommunicator(application, f"/ws/game/{gid}/")
        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        await communicator.send_json_to({"strategies": ["search", "search"]})
        data = await communicator.receive_json_from()
        self.assertEqual(len(data["directions"]), 2)
        self.assertEqual(data["strategies"], ["search", "search"])
        self.assertEqual(len(data["events"]), 2)
        for d in data["directions"]:
            self.assertIn(d, ("up", "down", "left", "right"))
        self.assertEqual(len(data["state"]["snakes"]), 2)

        await communicator.disconnect()

    async def test_multiple_steps_over_one_connection(self):
        new = self.client.post("/api/new/", content_type="application/json").json()
        gid = new["game_id"]
        communicator = WebsocketCommunicator(application, f"/ws/game/{gid}/")
        await communicator.connect()

        # Two independent search AIs can legitimately end their duel early
        # (a collision), so this only checks that five request/response
        # round-trips over one connection work and steps never go backwards
        # -- not that the game necessarily lasts all five ticks.
        steps_history = []
        for _ in range(5):
            await communicator.send_json_to({"strategies": ["search", "search"]})
            data = await communicator.receive_json_from()
            steps_history.append(data["state"]["steps"])

        self.assertEqual(len(steps_history), 5)
        self.assertEqual(steps_history, sorted(steps_history))
        self.assertGreaterEqual(steps_history[0], 1)
        await communicator.disconnect()

    async def test_solo_game_steps_one_snake(self):
        new = self.client.post(
            "/api/new/", data=json.dumps({"strategies": ["search"]}),
            content_type="application/json",
        ).json()
        gid = new["game_id"]
        communicator = WebsocketCommunicator(application, f"/ws/game/{gid}/")
        await communicator.connect()

        await communicator.send_json_to({"strategies": ["search"]})
        data = await communicator.receive_json_from()
        self.assertEqual(len(data["directions"]), 1)
        self.assertEqual(data["strategies"], ["search"])
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(len(data["state"]["snakes"]), 1)
        self.assertIn(data["directions"][0], ("up", "down", "left", "right"))

        await communicator.disconnect()

    async def test_different_ai_per_snake_falls_back_independently(self):
        new = self.client.post("/api/new/", content_type="application/json").json()
        gid = new["game_id"]
        communicator = WebsocketCommunicator(application, f"/ws/game/{gid}/")
        await communicator.connect()

        await communicator.send_json_to({"strategies": ["search", "does-not-exist"]})
        data = await communicator.receive_json_from()
        # Snake 0 keeps its real strategy; snake 1's unknown one falls back.
        self.assertEqual(data["strategies"], ["search", "search"])

        await communicator.disconnect()

    async def test_unknown_game_id_sends_error_and_closes(self):
        # Well-formed (32 hex chars, matches the route) but no such game.
        communicator = WebsocketCommunicator(application, f"/ws/game/{'0' * 32}/")
        connected, _ = await communicator.connect()
        self.assertTrue(connected)  # the consumer accepts, then rejects

        data = await communicator.receive_json_from()
        self.assertEqual(data["error"], "unknown game_id")

        close = await communicator.receive_output()
        self.assertEqual(close["type"], "websocket.close")
        self.assertEqual(close.get("code"), 4404)

    async def test_malformed_game_id_is_refused_by_routing(self):
        # Doesn't match the route's 32-hex-char pattern at all: Channels'
        # URLRouter has no fallback to route to, so the connection is refused
        # before it ever reaches GameConsumer.
        communicator = WebsocketCommunicator(application, "/ws/game/not-a-valid-id/")
        with self.assertRaises(ValueError):
            await communicator.connect()
