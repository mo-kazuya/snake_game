"""WebSocket consumer for the per-step AI game loop.

Replaces the old ``POST /api/step/`` polling endpoint: the browser opens one
WebSocket connection per game (see ``game/routing.py`` for the URL) and sends
one small message per tick instead of a full HTTP request/response cycle.
``/api/new/`` and ``/api/state/`` stay plain HTTP — only the frequent
per-step traffic moves to the socket.

Uses the synchronous ``WebsocketConsumer`` (Channels runs each connection's
callbacks in a worker thread) so it can call straight into the existing
synchronous game code (``ai.choose``, ``GameState.step``, the in-memory
``store``) without an async rewrite.
"""

from __future__ import annotations

import json

from channels.generic.websocket import WebsocketConsumer

from . import ai, store


class GameConsumer(WebsocketConsumer):
    def connect(self):
        self.game_id = self.scope["url_route"]["kwargs"]["game_id"]
        self.accept()
        if store.get(self.game_id) is None:
            self._send_error_and_close()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data) if text_data else {}
        except ValueError:
            data = {}

        state = store.get(self.game_id)
        if state is None:
            self._send_error_and_close()
            return

        strategy = data.get("strategy", "search")
        direction, used = ai.choose(state, strategy)
        event = state.step(direction)
        self.send(text_data=json.dumps(
            {
                "game_id": self.game_id,
                "direction": direction,
                "strategy": used,
                "event": event,
                "state": state.to_dict(),
            }
        ))

    def _send_error_and_close(self):
        self.send(text_data=json.dumps({"error": "unknown game_id"}))
        self.close(code=4404)
