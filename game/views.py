"""HTTP endpoints for the AI-driven Snake game.

The browser is a thin renderer. All game state and all AI decisions live here
on the server:

    POST /api/new/    -> create a game, return its id and initial state
    GET  /api/state/  -> read the current state without advancing

The frequent per-tick move is *not* HTTP: once a game exists, the browser
opens a WebSocket to ``ws/game/<game_id>/`` (see ``game/consumers.py``) and
sends one small message per step instead of a full request/response cycle.
State is passed as JSON. The AI direction is computed in ``ai.choose_direction``.
"""

from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import rl_agent, store


def index(request):
    return render(request, "game/index.html")


def _parse_body(request) -> dict:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


_MAX_SNAKES = 2
_DEFAULT_STRATEGIES = ["search", "search"]


@csrf_exempt
@require_POST
def new_game(request):
    data = _parse_body(request)
    strategies = data.get("strategies", _DEFAULT_STRATEGIES)
    if not isinstance(strategies, list) or not strategies:
        strategies = _DEFAULT_STRATEGIES
    # One entry -> a solo game; two -> a battle. Extras beyond that are
    # dropped rather than spawning more snakes (untested board geometry).
    strategies = [s if isinstance(s, str) else "search" for s in strategies][:_MAX_SNAKES]

    # Board size: the client picks it freely (all current AIs are size-
    # independent); a model that requires a specific board would override it.
    try:
        grid = int(data.get("grid", 20))
    except (TypeError, ValueError):
        grid = 20
    for strategy in strategies:
        grid = rl_agent.required_grid(strategy) or grid
    grid = max(8, min(grid, 50))  # keep the board sane

    game_id, state = store.create(grid=grid, strategies=strategies)

    # Preload the selected RL policies in the background so their first move
    # isn't slow.
    for strategy in set(strategies):
        if strategy in rl_agent.MODEL_NAMES:
            rl_agent.warmup_async(strategy)

    return JsonResponse(
        {
            "game_id": game_id,
            "state": state.to_dict(),
            "strategies": rl_agent.strategies_meta(),
        }
    )


@require_GET
def state(request):
    game_id = request.GET.get("game_id")
    game = store.get(game_id) if game_id else None
    if game is None:
        return JsonResponse({"error": "unknown game_id"}, status=404)
    return JsonResponse({"game_id": game_id, "state": game.to_dict()})
