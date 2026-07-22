"""Give the ego/Transformer a *playstyle* with a LoRA adapter.

Same recipe as ``train_transformer_lora.py`` (freeze the base transformer, train
LoRA adapters -- FFN + heads + attention -- then merge to a drop-in checkpoint),
but the demonstrations come from a **playstyle-biased expert**
(:mod:`examples.style_experts`) instead of the balanced search AI. One adapter
per style, so a single base grows two personalities:

* ``aggressive`` — contests the shared food (takes it even when strictly ahead
  in a risky race) and crowds the opponent's head to cut off its space.
* ``defensive`` — yields contested food and repositions into open space away
  from the opponent, surviving longer.

Because each merged checkpoint is a plain ``EgoTransformer`` (identical
architecture), the Django adapter loads them through the same ``PPO.load`` path
-- exposed as the ``rl_trf_aggr`` and ``rl_trf_def`` strategies.

Run:

    python examples/train_transformer_style.py --style aggressive
    python examples/train_transformer_style.py --style defensive
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from examples.train_transformer_battle import MIX_SIZES  # noqa: E402
from examples.train_transformer_lora import (  # noqa: E402
    TARGET_SUFFIXES, inject_attn_lora, inject_lora, lora_bc,
    merge_to_clean_checkpoint,
)

GAMMA = 0.99
_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: style-biased expert demonstrations
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float, str]):
    """Play the styled expert as snake 0 vs a balanced search opponent until
    ``quota`` snake-0 transitions exist. Records ego obs (opponent folded in),
    the styled expert's relative action, and the base battle return."""
    seed, grid, quota, eps, style = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.envs.battle_env import _heading_index
    from gym_snake.obs import ego_observation

    from examples.style_experts import choose_direction_styled

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    ret_buf: list[float] = []

    def opp_cells(state, idx):
        return [
            cell for j, s in enumerate(state.snakes)
            if j != idx and s.alive for cell in s.body
        ]

    while len(act_buf) < quota:
        state = GameState(grid=grid)
        state.reset(["agent", "opponent"])
        ep_rewards: list[float] = []
        steps_since_food = 0

        while (not state.game_over and state.snakes[0].alive
               and steps_since_food < grid * grid):
            agent = state.snakes[0]
            hidx = _heading_index(agent.direction)
            obs = ego_observation(agent.body, state.food, grid, hidx,
                                  opponent_cells=opp_cells(state, 0))

            head, food = agent.body[0], state.food
            prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

            expert_dir = choose_direction_styled(state, 0, style)
            rel = {0: 0, 1: 1, 3: 2}[
                (_heading_index(expert_dir) - hidx) % 4]

            exec_dir = expert_dir
            if eps > 0 and random.random() < eps:
                other = opp_cells(state, 0)
                safe = [d for d in DIRECTIONS
                        if not _opposite(d, agent.direction)
                        and _simulate(agent.body, d, state.food, grid, set(other))]
                if safe:
                    exec_dir = random.choice(safe)

            opp_dir = state.snakes[1].direction
            if state.snakes[1].alive:
                opp_dir = choose_direction(state, 1)  # balanced opponent

            events = state.step([exec_dir, opp_dir])
            ev = events[0]

            reward = -0.005
            if ev["ate"]:
                reward += 1.0
                steps_since_food = 0
            elif ev["dead"] and not agent.stalled:
                reward += -1.0
            elif not ev["dead"]:
                nh = agent.body[0]
                nd = abs(nh[0] - food[0]) + abs(nh[1] - food[1])
                reward += 0.05 if nd < prev_dist else -0.05
                steps_since_food += 1

            obs_buf.append(obs.astype(np.float16))
            act_buf.append(rel)
            ep_rewards.append(reward)

        g = 0.0
        returns = np.empty(len(ep_rewards), dtype=np.float32)
        for i in range(len(ep_rewards) - 1, -1, -1):
            g = ep_rewards[i] + GAMMA * g
            returns[i] = g
        ret_buf.extend(returns.tolist())

    return (np.stack(obs_buf[:quota]).astype(np.float16),
            np.asarray(act_buf[:quota], dtype=np.int64),
            np.asarray(ret_buf[:quota], dtype=np.float32))


def collect_dataset(total, workers, out_path, style, eps=0.1):
    per_size = total // len(MIX_SIZES)
    chunk = 20_000
    tasks, seed = [], 0
    for grid in MIX_SIZES:
        remaining = per_size
        while remaining > 0:
            q = min(chunk, remaining)
            tasks.append((seed, grid, q, eps, style))
            seed += 1
            remaining -= q
    random.shuffle(tasks)
    print(f"[collect:{style}] {len(tasks)} chunks on {workers} workers", flush=True)
    t0 = time.time()
    op, ap, rp = [], [], []
    with mp.Pool(workers) as pool:
        for i, (o, a, r) in enumerate(pool.imap_unordered(_collect_chunk, tasks)):
            op.append(o); ap.append(a); rp.append(r)
            print(f"[collect:{style}] {i + 1}/{len(tasks)} "
                  f"({sum(len(x) for x in ap):,} transitions, {time.time() - t0:.0f}s)",
                  flush=True)
    obs, act, ret = np.concatenate(op), np.concatenate(ap), np.concatenate(rp)
    np.savez(out_path, obs=obs, act=act, ret=ret)
    print(f"[collect:{style}] saved {len(act):,} to {out_path} "
          f"(action dist {np.bincount(act)})", flush=True)
    return out_path


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", required=True, choices=["aggressive", "defensive"])
    p.add_argument("--init", type=Path,
                   default=REPO / "examples" / "ppo_snake_transformer.zip")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--transitions", type=int, default=48_000)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--no-attn", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_style"))
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or (REPO / "examples" / f"ppo_snake_transformer_{args.style[:4]}.zip")

    data_path = args.dataset
    if data_path is None or not data_path.exists():
        data_path = args.work_dir / f"{args.style}_demos.npz"
        if not data_path.exists():
            collect_dataset(args.transitions, args.workers, data_path, args.style)
        else:
            print(f"[collect:{args.style}] reusing {data_path}", flush=True)

    import torch
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] style={args.style} device={device} r={args.r} a={args.alpha}", flush=True)

    model = PPO.load(str(args.init), device=device)
    lora = inject_lora(model.policy, TARGET_SUFFIXES, args.r, args.alpha)
    if not args.no_attn:
        lora.update(inject_attn_lora(model.policy, args.r, args.alpha))

    lora_bc(model, lora, data_path, args.epochs, args.batch, args.lr, device)
    merge_to_clean_checkpoint(args.init, lora, device, out)
    print(f"[done:{args.style}] -> {out} ({out.stat().st_size // 1024} KiB)", flush=True)


if __name__ == "__main__":
    main()
