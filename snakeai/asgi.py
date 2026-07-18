"""
ASGI config for snakeai project.

It exposes the ASGI callable as a module-level variable named ``application``.
HTTP requests go to the regular Django app; WebSocket connections (used for
the per-step AI game loop, see ``game/consumers.py``) are routed separately.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "snakeai.settings")

# Must run before importing anything that touches Django models/apps (it's
# what populates the app registry).
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402

from game.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(URLRouter(websocket_urlpatterns)),
    }
)
