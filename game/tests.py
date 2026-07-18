"""Tests for the Snake engine, AI, and HTTP/WebSocket endpoints."""

from __future__ import annotations

import json
from collections import deque
from unittest import mock

from channels.testing import WebsocketCommunicator
from django.test import TestCase

from snakeai.asgi import application

from . import ai, rl_agent
from .engine import GameState


class EngineTests(TestCase):
    def test_initial_state(self):
        s = GameState(grid=20)
        self.assertEqual(len(s.snake), 3)
        self.assertEqual(s.direction, "right")
        self.assertFalse(s.game_over)
        # Food is never placed on the snake.
        self.assertNotIn(s.food, set(s.snake))

    def test_move_advances_head(self):
        s = GameState(grid=20)
        head = s.snake[0]
        s.step("right")
        self.assertEqual(s.snake[0], (head[0] + 1, head[1]))
        self.assertEqual(len(s.snake), 3)  # no growth without food

    def test_wall_collision_ends_game(self):
        s = GameState(grid=10)
        s.snake = [(9, 5), (8, 5), (7, 5)]
        s.direction = "right"
        event = s.step("right")
        self.assertTrue(event["dead"])
        self.assertTrue(s.game_over)

    def test_reversal_is_ignored(self):
        s = GameState(grid=20)
        s.direction = "right"
        s.step("left")  # 180-degree turn should be ignored
        self.assertEqual(s.direction, "right")

    def test_eating_grows_and_scores(self):
        s = GameState(grid=20)
        s.snake = [(5, 5), (4, 5), (3, 5)]
        s.direction = "right"
        s.food = (6, 5)
        event = s.step("right")
        self.assertTrue(event["ate"])
        self.assertEqual(s.score, 10)
        self.assertEqual(len(s.snake), 4)

    def test_serialization_roundtrip(self):
        s = GameState(grid=15)
        s.step("right")
        restored = GameState.from_dict(s.to_dict())
        self.assertEqual(restored.snake, s.snake)
        self.assertEqual(restored.food, s.food)
        self.assertEqual(restored.score, s.score)
        self.assertEqual(restored.steps_since_food, s.steps_since_food)
        self.assertEqual(restored.stalled, s.stalled)

    def test_stall_without_food_ends_game(self):
        """A move that doesn't reach food, once too many have piled up,
        must end the game even though it's perfectly legal (see ai.py's
        anti-loop tiebreaker for why this backstop exists)."""
        s = GameState(grid=10)
        s.snake = [(5, 5), (5, 6), (5, 7)]
        s.direction = "up"
        s.food = (0, 0)
        s.steps_since_food = s.grid * s.grid - 1  # one step from the limit
        event = s.step("left")
        self.assertTrue(event["moved"])
        self.assertFalse(event["ate"])
        self.assertTrue(event["dead"])
        self.assertTrue(s.game_over)
        self.assertTrue(s.stalled)

    def test_stall_flag_absent_on_normal_death(self):
        s = GameState(grid=10)
        s.snake = [(9, 5), (8, 5), (7, 5)]
        s.direction = "right"
        s.step("right")  # wall collision
        self.assertTrue(s.game_over)
        self.assertFalse(s.stalled)

    def test_eating_resets_stall_counter(self):
        s = GameState(grid=10)
        s.snake = [(5, 5), (4, 5), (3, 5)]
        s.direction = "right"
        s.food = (6, 5)
        s.steps_since_food = 50
        event = s.step("right")
        self.assertTrue(event["ate"])
        self.assertEqual(s.steps_since_food, 0)


class AITests(TestCase):
    def test_never_reverses(self):
        s = GameState(grid=20)
        s.direction = "right"
        d = ai.choose_direction(s)
        self.assertNotEqual(d, "left")

    def test_moves_toward_food(self):
        s = GameState(grid=20)
        s.snake = [(5, 5), (4, 5), (3, 5)]
        s.direction = "right"
        s.food = (5, 8)  # directly below
        # Straight to food is right/down; the AI should not pick a wasteful up.
        d = ai.choose_direction(s)
        self.assertIn(d, ("down", "right"))

    def test_avoids_immediate_death(self):
        # Snake hugging the right wall heading right -> must not step into wall.
        s = GameState(grid=8)
        s.snake = [(7, 3), (6, 3), (5, 3)]
        s.direction = "right"
        s.food = (7, 0)
        d = ai.choose_direction(s)
        self.assertNotEqual(d, "right")

    def test_ai_survives_many_steps(self):
        """A full AI-driven game should eat several foods without dying early."""
        s = GameState(grid=12)
        for _ in range(400):
            if s.game_over:
                break
            s.step(ai.choose_direction(s))
        # The safe-seeking AI should reach a decent score before any death.
        self.assertGreaterEqual(s.score, 30)

    def test_survival_tiebreak_avoids_recently_visited_cell(self):
        """Ties in survival mode should be broken by recency, not always the
        same fixed direction -- otherwise the AI can settle into repeating
        the exact same loop forever."""
        grid = 21
        mid = grid // 2
        s = GameState(grid=grid)
        s.snake = [(mid, mid), (mid, mid + 1), (mid, mid + 2)]  # heading "up"
        s.direction = "up"
        s.food = (0, 0)  # irrelevant here; _bfs is patched out below

        # Force branch 2 (survival) regardless of food placement: with no
        # safe path ever found, left/right/up are an exact tie by symmetry
        # on an open board.
        with mock.patch.object(ai, "_bfs", return_value=None):
            baseline = ai.choose_direction(s)

        landing = {
            "up": (mid, mid - 1),
            "left": (mid - 1, mid),
            "right": (mid + 1, mid),
        }[baseline]
        s._ai_recent_heads = deque([landing] * 10, maxlen=64)

        with mock.patch.object(ai, "_bfs", return_value=None):
            biased = ai.choose_direction(s)

        self.assertNotEqual(biased, baseline)


class RLAgentTests(TestCase):
    def _mirror(self, env):
        """Copy a gym SnakeEnv's live state into a Django GameState."""
        s = GameState(grid=env.grid_size)
        s.snake = list(env.snake)
        s.food = env.food
        s.direction = rl_agent._DIR_NAMES[env.heading_idx]
        return s

    def test_feature_observation_shape_and_range(self):
        s = GameState(grid=20)
        obs = rl_agent.build_observation(s, "features")
        self.assertEqual(obs.shape, (11,))
        self.assertTrue((obs >= 0).all() and (obs <= 1).all())

    def test_grid_observation_shape(self):
        s = GameState(grid=10)
        obs = rl_agent.build_observation(s, "grid")
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
            np.array_equal(rl_agent.build_observation(s, "features"), env._feature_obs())
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
            np.array_equal(rl_agent.build_observation(s, "grid"), env._grid_obs())
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
                np.array_equal(rl_agent.build_observation(s, "ego"), env._get_obs())
            )

    def test_required_grid(self):
        # Both RL models are board-size independent.
        self.assertIsNone(rl_agent.required_grid("rl"))
        self.assertIsNone(rl_agent.required_grid("rl_cnn"))

    def test_choose_falls_back_to_search_when_unknown(self):
        s = GameState(grid=20)
        direction, used = ai.choose(s, strategy="does-not-exist")
        self.assertEqual(used, "search")
        self.assertIn(direction, ("up", "down", "left", "right"))

    def test_rl_returns_valid_direction_when_available(self):
        if not rl_agent.is_available("rl"):
            self.skipTest("features model / stable-baselines3 not available")
        s = GameState(grid=20)
        direction, used = ai.choose(s, strategy="rl")
        self.assertEqual(used, "rl")
        self.assertNotEqual(direction, "left")  # never reverse (starts right)

    def test_cnn_works_on_any_board_size(self):
        if not rl_agent.is_available("rl_cnn"):
            self.skipTest("ego/CNN model not available")
        for grid in (10, 20):
            s = GameState(grid=grid)
            direction, used = ai.choose(s, strategy="rl_cnn")
            self.assertEqual(used, "rl_cnn", f"grid={grid}")
            self.assertIn(direction, ("up", "down", "right"))  # never reverse

    def test_transformer_works_on_any_board_size(self):
        if not rl_agent.is_available("rl_trf"):
            self.skipTest("ego/Transformer model not available")
        for grid in (10, 20):
            s = GameState(grid=grid)
            direction, used = ai.choose(s, strategy="rl_trf")
            self.assertEqual(used, "rl_trf", f"grid={grid}")
            self.assertIn(direction, ("up", "down", "right"))  # never reverse


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

    def test_new_game_accepts_various_grid_sizes(self):
        for grid in (10, 14, 30, 40):
            res = self.client.post(
                "/api/new/", data=json.dumps({"grid": grid}),
                content_type="application/json",
            )
            data = res.json()
            self.assertEqual(data["state"]["grid"], grid)
            # The snake starts mid-board with room to move on every size.
            head = data["state"]["snake"][0]
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

        await communicator.send_json_to({"strategy": "search"})
        data = await communicator.receive_json_from()
        self.assertIn(data["direction"], ("up", "down", "left", "right"))
        self.assertEqual(data["strategy"], "search")
        self.assertIn("state", data)

        await communicator.disconnect()

    async def test_multiple_steps_over_one_connection(self):
        new = self.client.post("/api/new/", content_type="application/json").json()
        gid = new["game_id"]
        communicator = WebsocketCommunicator(application, f"/ws/game/{gid}/")
        await communicator.connect()

        steps_seen = 0
        for _ in range(5):
            await communicator.send_json_to({"strategy": "search"})
            data = await communicator.receive_json_from()
            steps_seen = data["state"]["steps"]

        self.assertEqual(steps_seen, 5)
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
