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
* ``balanced`` — the base search AI's own ranking (a neutral middle style),
  cloned the same way so all three share one base + one LoRA slot.

Because each merged checkpoint is a plain ``EgoTransformer`` (identical
architecture), the Django adapter loads them through the same ``PPO.load`` path
-- exposed as the ``rl_trf_aggr`` and ``rl_trf_def`` strategies.

``--init`` picks the base the adapter is merged into, and ``--tag`` names the
result, so the same three styles can be grown on a stronger base: the
battle-trained checkpoint gives ``rl_trf_battle_aggr`` and friends.

Run:

    python examples/train_transformer_style.py --style aggressive
    python examples/train_transformer_style.py --style defensive

    # playstyles on top of the battle-trained base (GPU scale)
    python examples/train_transformer_style.py --style aggressive \\
        --init examples/ppo_snake_transformer_battle.zip --tag battle \\
        --transitions 300000 --epochs 8
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
    TARGET_SUFFIXES, base_action_logits, inject_attn_lora, inject_lora, lora_bc,
    merge_to_clean_checkpoint,
)

GAMMA = 0.99
_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: style-biased expert demonstrations
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float, str, int, bool]):
    """Play the styled expert as snake 0 vs a balanced search opponent until
    ``quota`` snake-0 transitions exist. Records ego obs (opponent folded in),
    the styled expert's relative action, and the base battle return."""
    seed, grid, quota, eps, style, window, opp_channels = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.envs.battle_env import _heading_index
    from gym_snake.obs import ego_observation

    from examples.style_experts import choose_direction_styled

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    ret_buf: list[float] = []

    def rivals(state, idx):
        return [s for j, s in enumerate(state.snakes) if j != idx and s.alive]

    def opp_cells(state, idx):
        return [cell for s in rivals(state, idx) for cell in s.body]

    while len(act_buf) < quota:
        state = GameState(grid=grid)
        state.reset(["agent", "opponent"])
        ep_rewards: list[float] = []
        steps_since_food = 0

        while (not state.game_over and state.snakes[0].alive
               and steps_since_food < grid * grid):
            agent = state.snakes[0]
            hidx = _heading_index(agent.direction)
            opp = rivals(state, 0)
            obs = ego_observation(agent.body, state.food, grid, hidx,
                                  window=window,
                                  opponent_cells=[c for r in opp for c in r.body],
                                  opponent_head=opp[0].body[0] if opp else None,
                                  opponent_channels=opp_channels)

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


def collect_dataset(total, workers, out_path, style, window=11,
                    opp_channels=False, eps=0.1):
    per_size = total // len(MIX_SIZES)
    chunk = 20_000
    tasks, seed = [], 0
    for grid in MIX_SIZES:
        remaining = per_size
        while remaining > 0:
            q = min(chunk, remaining)
            tasks.append((seed, grid, q, eps, style, window, opp_channels))
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
    p.add_argument("--style", required=True,
                   choices=["aggressive", "defensive", "balanced"])
    p.add_argument("--init", type=Path,
                   default=REPO / "examples" / "ppo_snake_transformer.zip",
                   help="base checkpoint the adapter is merged into")
    p.add_argument("--tag", default="",
                   help="name segment for the output, e.g. --tag battle -> "
                        "ppo_snake_transformer_battle_aggr.zip")
    p.add_argument("--window", type=int, default=11,
                   help="ego observation window; must match --init's window")
    p.add_argument("--opponent-channels", dest="opp_channels",
                   action="store_true",
                   help="build 8-channel observations (rival body/head); must "
                        "match --init. Auto-detected from --init when omitted")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--transitions", type=int, default=48_000)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--no-attn", action="store_true")
    p.add_argument("--anchor", type=float, default=0.0,
                   help="KL weight pulling the adapted policy back to --init on "
                        "every state (0 = plain BC). Use with a base that is "
                        "stronger than the styled expert, so the style is "
                        "learned as a delta instead of re-cloning the expert.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_style"))
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    from gym_snake.obs import check_window, ego_channels

    window = check_window(args.window)
    if not args.opp_channels and args.init.exists():
        # The base's observation space is the ground truth, so read it instead
        # of making the caller repeat it. (Reading the checkpoint's metadata is
        # cheap -- the weights stay on disk.)
        from stable_baselines3.common.save_util import load_from_zip_file

        base_data, _, _ = load_from_zip_file(str(args.init), device="cpu",
                                             load_data=True)
        base_shape = tuple(base_data["observation_space"].shape)
        args.opp_channels = base_shape[0] == ego_channels(True)
        if base_shape[1] != window:
            raise SystemExit(
                f"--init was trained with a {base_shape[1]}x{base_shape[1]} ego "
                f"window; pass --window {base_shape[1]}."
            )
    opp_channels = args.opp_channels
    _short = {"aggressive": "aggr", "defensive": "def", "balanced": "bal"}[args.style]
    stem = "_".join(x for x in ("ppo_snake_transformer", args.tag, _short) if x)
    out = args.out or (REPO / "examples" / f"{stem}.zip")

    data_path = args.dataset
    if data_path is None or not data_path.exists():
        # The demos depend on the style and the window only -- not on the base
        # checkpoint -- so keep them keyed by those and share across bases.
        ch = ego_channels(opp_channels)
        data_path = (args.work_dir /
                     f"{args.style}_w{window}c{ch}_{args.transitions // 1000}k.npz")
        if not data_path.exists():
            collect_dataset(args.transitions, args.workers, data_path,
                            args.style, window, opp_channels)
        else:
            print(f"[collect:{args.style}] reusing {data_path}", flush=True)

    import torch
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] style={args.style} base={args.init.name} device={device} "
          f"window={window} channels={ego_channels(opp_channels)} "
          f"r={args.r} a={args.alpha} anchor={args.anchor}", flush=True)

    model = PPO.load(str(args.init), device=device)
    expected = (ego_channels(opp_channels), window, window)
    if tuple(model.observation_space.shape) != expected:
        raise SystemExit(
            f"--init model expects observation {tuple(model.observation_space.shape)} "
            f"but this run builds {expected}: pass the matching --window / "
            f"--opponent-channels (both are part of the weights)."
        )
    # Snapshot the base's own answers *before* the adapters go in.
    base_logits = (base_action_logits(model, data_path, device)
                   if args.anchor > 0 else None)

    lora = inject_lora(model.policy, TARGET_SUFFIXES, args.r, args.alpha)
    if not args.no_attn:
        lora.update(inject_attn_lora(model.policy, args.r, args.alpha))

    lora_bc(model, lora, data_path, args.epochs, args.batch, args.lr, device,
            anchor=args.anchor, base_logits=base_logits)
    merge_to_clean_checkpoint(args.init, lora, device, out)
    print(f"[done:{args.style}] -> {out} ({out.stat().st_size // 1024} KiB)", flush=True)


if __name__ == "__main__":
    main()
