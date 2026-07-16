"""Tests for the Snake engine, AI, and HTTP endpoints."""

from __future__ import annotations

import json

from django.test import TestCase

from . import ai
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

    def test_step_advances_game(self):
        new = self.client.post("/api/new/", content_type="application/json").json()
        gid = new["game_id"]
        res = self.client.post(
            "/api/step/", data=json.dumps({"game_id": gid}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["direction"], ("up", "down", "left", "right"))
        self.assertIn("state", data)

    def test_unknown_game_id_404s(self):
        res = self.client.post(
            "/api/step/", data=json.dumps({"game_id": "nope"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 404)

    def test_index_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "AIスネークゲーム")
