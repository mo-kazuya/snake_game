"""HTTP endpoints for the AI-driven Snake game.

The browser is a thin renderer. All game state and all AI decisions live here
on the server:

    POST /api/new/    -> create a game, return its id and initial state
    POST /api/step/   -> let the AI make one move, return the new state
    GET  /api/state/  -> read the current state without advancing

State is passed as JSON. The AI direction is computed in ``ai.choose_direction``.
"""

from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import ai, rl_agent, store


def index(request):
    return render(request, "game/index.html")


def _parse_body(request) -> dict:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


@csrf_exempt
@require_POST
def new_game(request):
    data = _parse_body(request)
    grid = int(data.get("grid", 20))
    grid = max(8, min(grid, 50))  # keep the board sane
    game_id, state = store.create(grid=grid)
    rl_available = rl_agent.is_available()
    if rl_available:
        # Preload the policy in the background so the first RL move isn't slow.
        rl_agent.warmup_async()
    return JsonResponse(
        {
            "game_id": game_id,
            "state": state.to_dict(),
            "rl_available": rl_available,
        }
    )


@csrf_exempt
@require_POST
def step(request):
    data = _parse_body(request)
    game_id = data.get("game_id")
    state = store.get(game_id) if game_id else None
    if state is None:
        return JsonResponse({"error": "unknown game_id"}, status=404)

    strategy = data.get("strategy", "search")
    direction, used = ai.choose(state, strategy)
    event = state.step(direction)
    return JsonResponse(
        {
            "game_id": game_id,
            "direction": direction,
            "strategy": used,
            "event": event,
            "state": state.to_dict(),
        }
    )


@require_GET
def state(request):
    game_id = request.GET.get("game_id")
    game = store.get(game_id) if game_id else None
    if game is None:
        return JsonResponse({"error": "unknown game_id"}, status=404)
    return JsonResponse({"game_id": game_id, "state": game.to_dict()})
