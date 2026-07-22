"""Fine-tune the ego/Transformer for **two-snake battle mode**.

The shipped ego/Transformer (``examples/ppo_snake_transformer.zip``) is strong
but was trained **single-snake**: in the Django battle it only sees the opponent
folded into its danger channels as a static hazard, and it never learned to
contest the shared food, dodge a *moving* rival, or avoid a head-on crash. This
script produces a battle-specialized sibling, ``ppo_snake_transformer_battle.zip``,
that the Django adapter exposes as the ``rl_trf_battle`` strategy.

Pipeline (same shape as ``train_transformer_v2.py``, but everything happens on
the two-snake :class:`~gym_snake.envs.battle_env.SnakeBattleEnv`, whose rules are
the exact engine the Django server runs):

1. **Warm-start** from the shipped single-snake Transformer (``--init``). Its
   observation space ``(5, 11, 11)`` and action space ``Discrete(3)`` are
   identical to the battle env, so the weights transfer directly — no surgery.
   (Use ``--no-init`` to train the same architecture from scratch instead.)
2. *(optional)* **Behavior-clone** on expert **battle** demonstrations: the BFS
   search AI playing snake 0 against a search-AI opponent, so the value head and
   policy get an opponent-aware warm start before RL. Enable with
   ``--bc-transitions N`` (0 = skip, the default when ``--init`` supplies a
   ready policy).
3. **PPO fine-tune** on a ``SubprocVecEnv`` mixing board sizes *and* opponents
   (BFS search + random-safe by default), with a "best by battle eval"
   checkpoint gate (mean food AND win-rate vs the search AI).

**Self-play across generations:** pass a previous checkpoint as an opponent,
e.g. ``--opponents search,model:examples/ppo_snake_transformer_battle.zip`` on a
second run, to fine-tune against your own prior best.

Run:

    # warm-start from the shipped model, PPO fine-tune vs search+random
    python examples/train_transformer_battle.py

    # add an opponent-aware BC warm start first
    python examples/train_transformer_battle.py --bc-transitions 600000

    # self-play against a previous checkpoint
    python examples/train_transformer_battle.py \
        --opponents search,model:examples/ppo_snake_transformer_battle.zip

The result stays loadable by the Django adapter (``PPO.load(..., device="cpu")``)
because the ``EgoTransformer`` class and its kwargs live in the checkpoint.
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
sys.path.insert(0, str(REPO))  # for `game.*` (gym_snake is pip-installed)

# Board sizes mixed during collection, PPO fine-tuning and evaluation.
MIX_SIZES = [8, 10, 12, 14, 16, 20, 24, 30, 40]
EVAL_SIZES = [8, 10, 14, 20, 30, 40]

GAMMA = 0.99

# Same architecture as the shipped v2 Transformer, so --init weights load
# straight in (and --no-init reproduces that architecture from scratch).
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4,
                    dim_feedforward=512, patch_size=1)

# Battle-specific reward bonuses layered on top of the SnakeEnv-identical base.
REWARD_OPP_DEATH = 0.5   # once, when the opponent dies while we're alive
REWARD_WIN = 1.0         # terminal, if our score > opponent's
REWARD_LOSE = -1.0       # terminal, if our score < opponent's

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: expert *battle* demonstrations (search AI vs search AI, parallel)
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float]):
    """Play the search AI as snake 0 against a search-AI opponent on `grid`
    boards until `quota` transitions for snake 0 exist.

    The recorded observation is snake 0's egocentric view **with the live
    opponent body folded in** (``opponent_cells``), the label is the expert's
    relative action, and the return uses the battle env's base reward scheme
    (used to warm-start the value head). With probability `eps` a random *safe*
    move is executed instead of the expert's (DAgger-style off-distribution
    coverage) but the label stays the expert's.
    """
    seed, grid, quota, eps = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.envs.battle_env import _heading_index
    from gym_snake.obs import ego_observation

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    ret_buf: list[float] = []

    def opp_cells(state, idx):
        return [
            cell
            for j, s in enumerate(state.snakes)
            if j != idx and s.alive
            for cell in s.body
        ]

    while len(act_buf) < quota:
        state = GameState(grid=grid)
        state.reset(["agent", "opponent"])
        ep_rewards: list[float] = []
        steps_since_food = 0

        while not state.game_over and state.snakes[0].alive and steps_since_food < grid * grid:
            agent = state.snakes[0]
            hidx = _heading_index(agent.direction)
            obs = ego_observation(agent.body, state.food, grid, hidx,
                                  opponent_cells=opp_cells(state, 0))

            head, food = agent.body[0], state.food
            prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

            expert_dir = choose_direction(state, 0)
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
                opp_dir = choose_direction(state, 1)

            events = state.step([exec_dir, opp_dir])
            ev = events[0]

            # Reward, matching SnakeBattleEnv's base scheme exactly.
            reward = -0.005  # REWARD_STEP
            if ev["ate"]:
                reward += 1.0  # REWARD_FOOD
                steps_since_food = 0
            elif ev["dead"] and not agent.stalled:
                reward += -1.0  # REWARD_DEATH (stall -> no penalty, like the env)
            elif not ev["dead"]:
                new_head = agent.body[0]
                new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
                reward += 0.05 if new_dist < prev_dist else -0.05
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

    obs_arr = np.stack(obs_buf[:quota]).astype(np.float16)
    act_arr = np.asarray(act_buf[:quota], dtype=np.int64)
    ret_arr = np.asarray(ret_buf[:quota], dtype=np.float32)
    return obs_arr, act_arr, ret_arr


def collect_dataset(total: int, workers: int, out_path: Path,
                    eps: float = 0.1) -> Path:
    per_size = total // len(MIX_SIZES)
    chunk = 20_000
    tasks = []
    seed = 0
    for grid in MIX_SIZES:
        remaining = per_size
        while remaining > 0:
            q = min(chunk, remaining)
            tasks.append((seed, grid, q, eps))
            seed += 1
            remaining -= q
    random.shuffle(tasks)

    print(f"[collect] {len(tasks)} chunks on {workers} workers "
          f"({per_size:,} transitions x {len(MIX_SIZES)} sizes)", flush=True)
    t0 = time.time()
    obs_parts, act_parts, ret_parts = [], [], []
    with mp.Pool(workers) as pool:
        for i, (o, a, r) in enumerate(pool.imap_unordered(_collect_chunk, tasks)):
            obs_parts.append(o)
            act_parts.append(a)
            ret_parts.append(r)
            done = sum(len(x) for x in act_parts)
            print(f"[collect] chunk {i + 1}/{len(tasks)} done "
                  f"({done:,} transitions, {time.time() - t0:.0f}s)", flush=True)

    obs = np.concatenate(obs_parts)
    act = np.concatenate(act_parts)
    ret = np.concatenate(ret_parts)
    np.savez(out_path, obs=obs, act=act, ret=ret)
    print(f"[collect] saved {len(act):,} transitions to {out_path} "
          f"({time.time() - t0:.0f}s). action dist: {np.bincount(act)}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def make_env_fn(grid: int, opponent: str):
    def _thunk():
        import gymnasium as gym

        import gym_snake  # noqa: F401

        return gym.make(
            "gym_snake/SnakeBattle-v0", grid_size=grid, opponent=opponent,
            reward_opp_death=REWARD_OPP_DEATH,
            reward_win=REWARD_WIN, reward_lose=REWARD_LOSE,
        )

    return _thunk


def evaluate(model, sizes, episodes: int, opponent: str = "search",
             seed0: int = 1000) -> tuple[dict[int, float], dict[int, float]]:
    """Greedy eval vs a fixed opponent. Returns (mean food, win-rate) per size."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    food: dict[int, float] = {}
    winrate: dict[int, float] = {}
    for grid in sizes:
        vec = DummyVecEnv([make_env_fn(grid, opponent) for _ in range(episodes)])
        vec.seed(seed0)
        obs = vec.reset()
        finished = np.zeros(episodes, dtype=bool)
        eps_food = np.zeros(episodes)
        eps_win = np.zeros(episodes)
        while not finished.all():
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(actions)
            for i in range(episodes):
                if dones[i] and not finished[i]:
                    finished[i] = True
                    eps_food[i] = infos[i]["score"]
                    eps_win[i] = 1.0 if infos[i].get("win") else 0.0
        vec.close()
        food[grid] = float(eps_food.mean())
        winrate[grid] = float(eps_win.mean())
    return food, winrate


# ---------------------------------------------------------------------------
# Phase 2: behavior cloning (identical recipe to train_transformer_v2)
# ---------------------------------------------------------------------------

def behavior_clone(model, data_path: Path, epochs: int, batch_size: int,
                   lr: float, device) -> None:
    import torch
    import torch.nn.functional as F

    data = np.load(data_path)
    obs_all = torch.from_numpy(data["obs"])           # (N,5,11,11) float16
    act_all = torch.from_numpy(data["act"])           # (N,) int64
    ret_all = torch.from_numpy(data["ret"])           # (N,) float32
    n = len(act_all)
    n_val = max(2048, n // 50)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[bc] {len(train_idx):,} train / {n_val:,} val transitions", flush=True)

    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    total_steps = epochs * (len(train_idx) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.05)

    def mirror(obs_b, act_b):
        """Flip half of each batch left/right (ego frame): swap turn L/R."""
        m = torch.rand(len(act_b), device=obs_b.device) < 0.5
        obs_b = torch.where(m.view(-1, 1, 1, 1), obs_b.flip(-1), obs_b)
        swapped = torch.where(act_b == 1, torch.full_like(act_b, 2),
                              torch.where(act_b == 2, torch.full_like(act_b, 1), act_b))
        return obs_b, torch.where(m, swapped, act_b)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(len(train_idx))]
        tot_ce = tot_vf = tot_n = 0
        for i in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            obs_b = obs_all[idx].to(device, torch.float32)
            act_b = act_all[idx].to(device)
            ret_b = ret_all[idx].to(device)
            obs_b, act_b = mirror(obs_b, act_b)

            values, log_prob, entropy = policy.evaluate_actions(obs_b, act_b)
            ce = -log_prob.mean()
            vf = F.mse_loss(values.flatten(), ret_b)
            loss = ce + 0.5 * vf - 0.003 * entropy.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_ce += ce.item() * len(idx)
            tot_vf += vf.item() * len(idx)
            tot_n += len(idx)

        policy.set_training_mode(False)
        with torch.no_grad():
            correct = ce_sum = 0.0
            for i in range(0, n_val, batch_size):
                idx = val_idx[i:i + batch_size]
                obs_b = obs_all[idx].to(device, torch.float32)
                act_b = act_all[idx].to(device)
                dist = policy.get_distribution(obs_b)
                logits = dist.distribution.logits
                ce_sum += F.cross_entropy(logits, act_b, reduction="sum").item()
                correct += (logits.argmax(-1) == act_b).sum().item()
        policy.set_training_mode(True)
        print(f"[bc] epoch {epoch}/{epochs}: train ce={tot_ce / tot_n:.4f} "
              f"vf={tot_vf / tot_n:.3f} | val ce={ce_sum / n_val:.4f} "
              f"acc={correct / n_val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    policy.set_training_mode(False)


# ---------------------------------------------------------------------------
# Phase 3: PPO fine-tuning with a best-by-battle-eval gate
# ---------------------------------------------------------------------------

def ppo_finetune(model, timesteps: int, best_path: Path, eval_every: int) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class BestGate(BaseCallback):
        """Every `eval_every` steps evaluate vs the search AI on small/mid/large
        boards and keep the checkpoint with the best combined score (mean food
        plus a win-rate bonus), so the gate rewards *winning*, not just eating."""

        gate_sizes = (10, 20, 40)

        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.next_eval = eval_every

        def _eval_and_save(self):
            food, winrate = evaluate(self.model, self.gate_sizes, episodes=5)
            mean_food = float(np.mean(list(food.values())))
            mean_win = float(np.mean(list(winrate.values())))
            combined = mean_food + 10.0 * mean_win  # win-rate weighted in
            line = " ".join(f"{g}:{food[g]:.1f}/{winrate[g]:.0%}" for g in self.gate_sizes)
            marker = ""
            if combined > self.best:
                self.best = combined
                self.model.save(str(best_path))
                marker = "  <- new best, saved"
            print(f"[gate] steps={self.num_timesteps:,} food={mean_food:.2f} "
                  f"win={mean_win:.0%} ({line}){marker}", flush=True)

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                self._eval_and_save()
            return True

        def _on_training_end(self) -> None:
            self._eval_and_save()

    model.learn(total_timesteps=timesteps, callback=BestGate(),
                reset_num_timesteps=True, progress_bar=False)


# ---------------------------------------------------------------------------

def build_model(vec_env, init_path: Path | None, device):
    """Build a fresh battle PPO and (optionally) warm-start its **policy
    weights** from ``init_path``.

    Rather than ``PPO.load``-ing the old checkpoint and overriding its stored
    hyperparameters (which would clobber SB3's internal ``clip_range`` /
    ``learning_rate`` *schedule* objects with plain floats), we construct a
    fresh PPO with the correct battle hyperparameters — so every schedule is set
    up properly — and copy only the network ``state_dict`` across. This is safe
    precisely because the battle env's observation/action spaces and the
    ``EgoTransformer`` architecture (``MODEL_KWARGS``) are identical to the
    shipped single-snake model, so the keys line up exactly.
    """
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401  (registers EgoTransformer)
    from gym_snake.policies import transformer_policy_kwargs

    model = PPO(
        "CnnPolicy", vec_env, verbose=1, device=device,
        n_steps=1024, batch_size=1024, n_epochs=4,
        gamma=GAMMA, gae_lambda=0.95, ent_coef=0.005, clip_range=0.15,
        learning_rate=lambda pr: 1e-4 * pr,
        policy_kwargs=transformer_policy_kwargs(**MODEL_KWARGS),
    )

    if init_path is not None and init_path.exists():
        print(f"[setup] warm-starting policy weights from {init_path}", flush=True)
        source = PPO.load(str(init_path), device=device)
        missing, unexpected = model.policy.load_state_dict(
            source.policy.state_dict(), strict=False)
        if missing or unexpected:
            print(f"[setup] WARNING: state_dict mismatch "
                  f"(missing={list(missing)}, unexpected={list(unexpected)})",
                  flush=True)
        else:
            print("[setup] warm-start weights loaded (all keys matched)", flush=True)
        del source
    else:
        print("[setup] no init model -> training EgoTransformer from scratch",
              flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer.zip",
                        help="single-snake Transformer to warm-start from")
    parser.add_argument("--no-init", action="store_true",
                        help="ignore --init and train the architecture from scratch")
    parser.add_argument("--opponents", type=str, default="search,random",
                        help="comma-separated opponent specs cycled across the "
                             "parallel envs: search | random | model:<path>")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="reuse an existing .npz battle-demo dataset")
    parser.add_argument("--bc-transitions", type=int, default=0,
                        help="battle demos to collect+clone before PPO (0=skip)")
    parser.add_argument("--demo-eps", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer_battle.zip")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_battle"))
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    init_path = None if args.no_init else args.init

    # -- optional battle-demo dataset ---------------------------------------
    data_path = args.dataset
    if args.bc_transitions > 0 and data_path is None:
        data_path = args.work_dir / "battle_demos.npz"
        if not data_path.exists():
            collect_dataset(args.bc_transitions, args.workers, data_path,
                            eps=args.demo_eps)
        else:
            print(f"[collect] reusing cached {data_path}", flush=True)

    # -- model + envs -------------------------------------------------------
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    sizes = [MIX_SIZES[i % len(MIX_SIZES)] for i in range(args.n_envs)]
    env_fns = [make_env_fn(sizes[i], opponents[i % len(opponents)])
               for i in range(args.n_envs)]
    vec_env = VecMonitor(SubprocVecEnv(env_fns))
    print(f"[setup] {args.n_envs} envs, sizes={sizes}, opponents={opponents}", flush=True)

    model = build_model(vec_env, init_path, device)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[setup] policy parameters: {n_params:,}", flush=True)

    # -- optional behavior cloning ------------------------------------------
    if data_path is not None:
        behavior_clone(model, data_path, args.bc_epochs, args.bc_batch,
                       args.bc_lr, model.device)
        bc_path = args.work_dir / "battle_bc.zip"
        model.save(str(bc_path))
        food, win = evaluate(model, EVAL_SIZES, episodes=10)
        print("[bc] eval food:", {g: round(v, 2) for g, v in food.items()},
              "win:", {g: round(v, 2) for g, v in win.items()}, flush=True)

    if args.skip_ppo:
        model.save(str(args.out))
        vec_env.close()
        print(f"[done] saved (no PPO) to {args.out}", flush=True)
        return

    # -- PPO fine-tune ------------------------------------------------------
    best_path = args.work_dir / "battle_ppo_best.zip"
    ppo_finetune(model, args.timesteps, best_path, args.eval_every)
    vec_env.close()

    # -- final: pick best gate checkpoint, evaluate thoroughly --------------
    from stable_baselines3 import PPO

    final = PPO.load(str(best_path if best_path.exists() else args.out
                         if args.out.exists() else best_path), device=device)
    food, win = evaluate(final, EVAL_SIZES, episodes=20)
    print("[final] food (20 eps/size):", {g: round(v, 2) for g, v in food.items()}, flush=True)
    print("[final] win-rate vs search:", {g: round(v, 2) for g, v in win.items()}, flush=True)
    final.save(str(args.out))
    print(f"[final] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
