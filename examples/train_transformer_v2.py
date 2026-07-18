"""Retrain the ego/Transformer agent: imitation learning + PPO fine-tuning.

Pipeline (GPU + many CPU cores):

1. **Collect** expert demonstrations from the search AI (``game.ai``) on a mix
   of board sizes, in parallel worker processes. Each transition stores the
   ego observation, the expert's relative action, and the discounted return
   under the SnakeEnv reward scheme (used to warm-start the value head).
2. **Behavior-clone** a *larger* EgoTransformer (d_model=128, 8 heads,
   4 layers, FFN 512, per-cell patch_size=1) on the GPU: cross-entropy on the
   expert actions + MSE on the returns, with left/right mirror augmentation.
3. **PPO fine-tune** on a SubprocVecEnv mixing the same board sizes, keeping a
   "best by mixed-size eval" checkpoint gate like the CNN rounds did.

Run:

    python examples/train_transformer_v2.py                # full pipeline
    python examples/train_transformer_v2.py --dataset D.npz --timesteps 8000000

The final model is written to ``--out`` (default
``examples/ppo_snake_transformer_v2.zip``) and stays loadable by the Django
adapter (`PPO.load(..., device="cpu")`) because the EgoTransformer class and
its kwargs are stored in the checkpoint.
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

# Larger EgoTransformer than the shipped v1 (d64/2 layers/FFN128/patch2).
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4,
                    dim_feedforward=512, patch_size=1)

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: expert data collection (search AI, parallel workers)
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float]):
    """Play the search AI on `grid` boards until `quota` transitions exist.

    With probability `eps` per step, a random *safe* move is executed instead
    of the expert's — but the recorded label is always the expert's choice
    (DAgger-style noise injection, so the policy also learns how the expert
    recovers from slightly off-distribution states).
    """
    seed, grid, quota, eps = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.obs import ego_observation

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    ret_buf: list[float] = []

    while len(act_buf) < quota:
        state = GameState(grid=grid)
        ep_rewards: list[float] = []
        ep_start = len(act_buf)
        steps_since_food = 0

        while not state.game_over and steps_since_food < grid * grid:
            hidx = _HEADINGS.index(DIRECTIONS[state.direction])
            obs = ego_observation(state.snake, state.food, grid, hidx)

            head, food = state.snake[0], state.food
            prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

            expert_dir = choose_direction(state)
            rel = {0: 0, 1: 1, 3: 2}[
                (_HEADINGS.index(DIRECTIONS[expert_dir]) - hidx) % 4]

            exec_dir = expert_dir
            if eps > 0 and random.random() < eps:
                safe = [d for d in DIRECTIONS
                        if not _opposite(d, state.direction)
                        and _simulate(state.snake, d, state.food, grid)]
                if safe:
                    exec_dir = random.choice(safe)
            event = state.step(exec_dir)

            # Reward, matching SnakeEnv exactly.
            reward = -0.005  # REWARD_STEP
            if event["ate"]:
                reward += 1.0  # REWARD_FOOD (also the board-full "win" case)
                steps_since_food = 0
            elif event["dead"]:
                reward += -1.0  # REWARD_DEATH
            else:
                new_head = state.snake[0]
                new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
                reward += 0.05 if new_dist < prev_dist else -0.05  # shaping
                steps_since_food += 1

            obs_buf.append(obs.astype(np.float16))
            act_buf.append(rel)
            ep_rewards.append(reward)

        # Discounted returns for this episode (0 bootstrap at the end).
        g = 0.0
        returns = np.empty(len(ep_rewards), dtype=np.float32)
        for i in range(len(ep_rewards) - 1, -1, -1):
            g = ep_rewards[i] + GAMMA * g
            returns[i] = g
        ret_buf.extend(returns.tolist())
        assert len(ret_buf) == len(act_buf), (len(ret_buf), len(act_buf), ep_start)

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

def make_env_fn(grid: int):
    def _thunk():
        import gymnasium as gym

        import gym_snake  # noqa: F401

        return gym.make("gym_snake/Snake-v0", grid_size=grid, obs_type="ego")

    return _thunk


def evaluate(model, sizes, episodes: int, seed0: int = 1000) -> dict[int, float]:
    """Mean greedy score per board size (all episodes stepped as one batch)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    scores: dict[int, float] = {}
    for grid in sizes:
        vec = DummyVecEnv([make_env_fn(grid) for _ in range(episodes)])
        vec.seed(seed0)
        obs = vec.reset()
        finished = np.zeros(episodes, dtype=bool)
        eps_scores = np.zeros(episodes)
        while not finished.all():
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(actions)
            for i in range(episodes):
                if dones[i] and not finished[i]:
                    finished[i] = True
                    eps_scores[i] = infos[i]["score"]
        vec.close()
        scores[grid] = float(eps_scores.mean())
    return scores


# ---------------------------------------------------------------------------
# Phase 2: behavior cloning
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

        # Validation
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
# Phase 3: PPO fine-tuning with a best-model gate
# ---------------------------------------------------------------------------

def ppo_finetune(model, timesteps: int, best_path: Path, eval_every: int) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class BestGate(BaseCallback):
        """Every `eval_every` steps, evaluate on small/mid/large boards and
        keep the checkpoint with the best mean score."""

        gate_sizes = (10, 20, 40)

        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.next_eval = eval_every

        def _eval_and_save(self):
            scores = evaluate(self.model, self.gate_sizes, episodes=5)
            mean = float(np.mean(list(scores.values())))
            line = " ".join(f"{g}:{s:.1f}" for g, s in scores.items())
            marker = ""
            if mean > self.best:
                self.best = mean
                self.model.save(str(best_path))
                marker = "  <- new best, saved"
            print(f"[gate] steps={self.num_timesteps:,} mean={mean:.2f} "
                  f"({line}){marker}", flush=True)

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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="reuse an existing .npz demo dataset")
    parser.add_argument("--transitions", type=int, default=1_080_000)
    parser.add_argument("--demo-eps", type=float, default=0.1,
                        help="probability of executing a random safe move "
                             "during collection (labels stay the expert's)")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer_v2.zip")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_v2"))
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------
    data_path = args.dataset
    if data_path is None or not data_path.exists():
        data_path = args.work_dir / "expert_demos.npz"
        if not data_path.exists():
            collect_dataset(args.transitions, args.workers, data_path,
                            eps=args.demo_eps)
        else:
            print(f"[collect] reusing cached {data_path}", flush=True)

    # -- model --------------------------------------------------------------
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from gym_snake.policies import transformer_policy_kwargs

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    sizes = [MIX_SIZES[i % len(MIX_SIZES)] for i in range(args.n_envs)]
    vec_env = VecMonitor(SubprocVecEnv([make_env_fn(g) for g in sizes]))
    print(f"[setup] {args.n_envs} parallel envs, sizes={sizes}", flush=True)

    def linear_decay(progress_remaining: float) -> float:
        return 1e-4 * progress_remaining

    model = PPO(
        "CnnPolicy", vec_env, verbose=1, device=device,
        n_steps=1024, batch_size=1024, n_epochs=4,
        gamma=GAMMA, gae_lambda=0.95, ent_coef=0.005, clip_range=0.15,
        learning_rate=linear_decay,
        policy_kwargs=transformer_policy_kwargs(**MODEL_KWARGS),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[setup] policy parameters: {n_params:,}", flush=True)

    # -- behavior cloning ---------------------------------------------------
    behavior_clone(model, data_path, args.bc_epochs, args.bc_batch,
                   args.bc_lr, model.device)
    bc_path = args.work_dir / "transformer_bc.zip"
    model.save(str(bc_path))
    print(f"[bc] saved BC model to {bc_path}", flush=True)

    bc_scores = evaluate(model, EVAL_SIZES, episodes=10)
    print("[bc] eval:", {g: round(s, 2) for g, s in bc_scores.items()}, flush=True)

    if args.skip_ppo:
        model.save(str(args.out))
        vec_env.close()
        return

    # -- PPO fine-tune ------------------------------------------------------
    best_path = args.work_dir / "transformer_ppo_best.zip"
    ppo_finetune(model, args.timesteps, best_path, args.eval_every)
    vec_env.close()

    # -- final: pick best gate checkpoint, evaluate thoroughly --------------
    final = PPO.load(str(best_path if best_path.exists() else bc_path),
                     device=device)
    final_scores = evaluate(final, EVAL_SIZES, episodes=20)
    print("[final] eval (20 eps/size):",
          {g: round(s, 2) for g, s in final_scores.items()}, flush=True)
    final.save(str(args.out))
    print(f"[final] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
