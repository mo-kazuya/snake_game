"""Playstyle-biased expert policies, used to generate BC demonstrations.

The shipped search AI (:func:`game.ai.choose_direction`) plays a balanced,
survival-first game. To give a LoRA adapter a *personality*, we clone a
demonstration expert whose move ranking is biased toward a style, then behavior-
clone it. Two styles:

* ``"aggressive"`` — contest the food hard (take any safe food, even when
  slightly behind in the race) and, when not eating, **stay close to the
  opponent's head** to crowd it and cut off its space.
* ``"defensive"`` — only take food that is safe *and* uncontested (don't lose a
  risky race), and otherwise **maximize open space while keeping distance from
  the opponent** to outlast it.

Both reuse the exact geometry helpers of :mod:`game.ai` (BFS path, tail
reachability, flood-fill openness, opponent bodies as obstacles), so they stay
consistent with how the real game scores safety.
"""

from __future__ import annotations

from game.ai import (
    _bfs, _cell_to_direction, _flood_fill_size, _opposite, _other_bodies,
    _reachable_tail, _record_head, _simulate,
)
from game.engine import DIRECTIONS


def _opponent_head(state, idx):
    for j, s in enumerate(state.snakes):
        if j != idx and s.alive and s.body:
            return s.body[0]
    return None


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_direction_styled(state, idx: int, style: str) -> str:
    """Pick a direction for ``state.snakes[idx]`` with a playstyle bias.

    ``style`` is ``"aggressive"``, ``"defensive"`` or ``"balanced"`` (the
    latter reproduces the base search AI's ranking).
    """
    grid = state.grid
    me = state.snakes[idx]
    snake = me.body
    head = snake[0]
    food = state.food
    recent = _record_head(state, idx)
    other = _other_bodies(state, idx)
    opp = _opponent_head(state, idx)

    legal = [
        d for d in DIRECTIONS
        if not _opposite(d, me.direction) and _simulate(snake, d, food, grid, other)
    ]
    if not legal:
        return me.direction

    my_food = _manhattan(head, food) if food else 10 ** 9
    opp_food = _manhattan(opp, food) if (opp and food) else 10 ** 9
    # "Contested": the opponent is at least as close to the food as we are, so
    # heading straight for it risks a lost race or a head-on. This is where the
    # two styles diverge -- aggressive charges, defensive yields.
    contested = opp is not None and food is not None and opp_food <= my_food

    # -- food seeking, gated by style -------------------------------------
    body_blocked = set(snake[:-1]) | other
    path = _bfs(head, food, body_blocked, grid) if food else None
    if path:
        move = _cell_to_direction(head, path[0])
        if move in legal:
            after = _simulate(snake, move, food, grid, other)
            safe = bool(after) and _reachable_tail(after, food, grid, other)
            if style == "aggressive":
                if safe:
                    return move                       # always contest safe food
                # Risk-take: grab a contested food only when we are strictly
                # ahead in the race (avoids the tied head-on that just trades
                # both snakes), even without a tail guarantee, to deny it.
                if after and my_food < opp_food:
                    return move
            elif style == "defensive":
                # Take food only when it is *uncontested* and the path is safe
                # and leaves room to live. When the opponent is racing us for
                # it, yield and reposition (fall through to the survival ranking
                # below, which pulls us away from the rival).
                if safe and not contested:
                    blocked = set(after[1:-1]) | other
                    if _flood_fill_size(after[0], blocked, grid) >= len(snake):
                        return move
            else:  # balanced == base behavior
                if safe:
                    return move

    # -- survival / positioning, ranked by style --------------------------
    def openness(d) -> int:
        after = _simulate(snake, d, food, grid, other)
        if after is None:
            return -1
        blocked = set(after[1:-1]) | other
        return _flood_fill_size(after[0], blocked, grid)

    def opp_distance(d) -> int:
        after = _simulate(snake, d, food, grid, other)
        if after is None or opp is None:
            return 0
        return _manhattan(after[0], opp)

    def key(d):
        after = _simulate(snake, d, food, grid, other)
        tail_ok = 1 if (after and _reachable_tail(after, food, grid, other)) else 0
        fresh = 0 if (after and after[0] in recent) else 1
        opn = openness(d)
        od = opp_distance(d)
        if style == "aggressive":
            # Keep enough room to live (tail_ok, then openness), but among
            # comparable moves pull *toward* the opponent (smaller od first).
            return (tail_ok, opn, -od, fresh)
        if style == "defensive":
            # Survive first (openness), then actively keep away from the rival.
            return (tail_ok, opn, od, fresh)
        return (tail_ok, opn, fresh)

    return max(legal, key=key)
