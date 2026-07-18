"""WebSocket URL routing for the ``game`` app.

``game_id`` is a ``uuid4().hex`` string (32 lowercase hex chars), see
``game/store.py``.
"""

from __future__ import annotations

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r"^ws/game/(?P<game_id>[0-9a-f]{32})/$",
        consumers.GameConsumer.as_asgi(),
    ),
]
