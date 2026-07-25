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

Observation window (``--window``)
---------------------------------
The ego observation is a ``(5, W, W)`` tensor: a head-centered, heading-up
local view plus a whole-board minimap (``gym_snake/obs.py``). ``W`` used to be
hard-wired to 11; it is now a knob. A wider window sees more real geometry
around the head **and** a less lossy minimap, at a quadratic cost in
Transformer tokens (``(W / patch_size)**2``):

===========  ========  ==============  ===================
``--window``  tokens   BC throughput   CPU inference
===========  ========  ==============  ===================
11 (default)    121     22k samp/s        1.2 ms/step
21              441      5.3k samp/s      4.9 ms/step
===========  ========  ==============  ===================

Everything downstream follows the window automatically: ``SnakeEnv`` takes
``ego_window``, and the Django adapter reads a per-model ``window`` from its
registry. A model must always be *run* with the window it was *trained* with.

Speed
-----
* collection runs on all cores (``--workers``);
* the demo set is cached **on the GPU** when it fits (``--bc-cache``), which
  removes the host->device copy from every BC step;
* the Transformer trunk runs under ``bfloat16`` autocast on CUDA (``--amp``,
  on by default) in both BC and PPO — ~3x faster and ~2x less memory at 441
  tokens, while the policy/value heads and the PPO ratio math stay float32.

Run:

    python examples/train_transformer_v2.py                    # 11x11, as shipped
    python examples/train_transformer_v2.py --window 21        # wide view
    python examples/train_transformer_v2.py --dataset D.npz --timesteps 8000000

The final model is written to ``--out`` (default
``examples/ppo_snake_transformer_v2.zip``, or ``..._w<W>.zip`` for a non-default
window) and stays loadable by the Django adapter
(`PPO.load(..., device="cpu")`) because the EgoTransformer class and its kwargs
are stored in the checkpoint.
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
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4, dim_feedforward=512)

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: expert data collection (search AI, parallel workers)
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float, int]):
    """Play the search AI on `grid` boards until `quota` transitions exist.

    With probability `eps` per step, a random *safe* move is executed instead
    of the expert's — but the recorded label is always the expert's choice
    (DAgger-style noise injection, so the policy also learns how the expert
    recovers from slightly off-distribution states).
    """
    seed, grid, quota, eps, window = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.obs import ego_observation

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    ret_buf: list[float] = []

    while len(act_buf) < quota:
        # One snake, spawned dead-center facing right -- the same starting
        # position SnakeEnv.reset() uses.
        state = GameState(grid=grid)
        state.reset(["agent"])
        me = state.snakes[0]
        ep_rewards: list[float] = []
        ep_start = len(act_buf)
        steps_since_food = 0

        while not state.game_over and me.alive and steps_since_food < grid * grid:
            hidx = _HEADINGS.index(DIRECTIONS[me.direction])
            obs = ego_observation(me.body, state.food, grid, hidx,
                                  window=window)

            head, food = me.body[0], state.food
            prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

            expert_dir = choose_direction(state, 0)
            rel = {0: 0, 1: 1, 3: 2}[
                (_HEADINGS.index(DIRECTIONS[expert_dir]) - hidx) % 4]

            exec_dir = expert_dir
            if eps > 0 and random.random() < eps:
                safe = [d for d in DIRECTIONS
                        if not _opposite(d, me.direction)
                        and _simulate(me.body, d, state.food, grid)]
                if safe:
                    exec_dir = random.choice(safe)
            event = state.step([exec_dir])[0]

            # Reward, matching SnakeEnv exactly.
            reward = -0.005  # REWARD_STEP
            if event["ate"]:
                reward += 1.0  # REWARD_FOOD (also the board-full "win" case)
                steps_since_food = 0
            elif event["dead"] and not me.stalled:
                # A stall is SnakeEnv's *truncation*, which carries no penalty.
                reward += -1.0  # REWARD_DEATH
            elif not event["dead"]:
                new_head = me.body[0]
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


def collect_dataset(total: int, workers: int, out_path: Path, window: int,
                    eps: float = 0.1) -> Path:
    per_size = total // len(MIX_SIZES)
    chunk = 20_000
    tasks = []
    seed = 0
    for grid in MIX_SIZES:
        remaining = per_size
        while remaining > 0:
            q = min(chunk, remaining)
            tasks.append((seed, grid, q, eps, window))
            seed += 1
            remaining -= q
    random.shuffle(tasks)

    # Fill one pre-allocated buffer instead of concatenating at the end: the
    # observations are ~4.8 GB at window 21, and np.concatenate would need a
    # second copy of that.
    n_total = sum(t[2] for t in tasks)
    obs = np.empty((n_total, 5, window, window), dtype=np.float16)
    act = np.empty(n_total, dtype=np.int64)
    ret = np.empty(n_total, dtype=np.float32)

    print(f"[collect] {len(tasks)} chunks on {workers} workers "
          f"({per_size:,} transitions x {len(MIX_SIZES)} sizes, window={window}, "
          f"{obs.nbytes / 2**30:.1f} GiB)", flush=True)
    t0 = time.time()
    at = 0
    with mp.Pool(workers) as pool:
        for i, (o, a, r) in enumerate(pool.imap_unordered(_collect_chunk, tasks)):
            obs[at:at + len(a)] = o
            act[at:at + len(a)] = a
            ret[at:at + len(a)] = r
            at += len(a)
            print(f"[collect] chunk {i + 1}/{len(tasks)} done "
                  f"({at:,} transitions, {time.time() - t0:.0f}s)", flush=True)

    np.savez(out_path, obs=obs[:at], act=act[:at], ret=ret[:at])
    print(f"[collect] saved {at:,} transitions to {out_path} "
          f"({time.time() - t0:.0f}s). action dist: {np.bincount(act[:at])}",
          flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def make_env_fn(grid: int, window: int):
    def _thunk():
        import gymnasium as gym

        import gym_snake  # noqa: F401

        return gym.make("gym_snake/Snake-v0", grid_size=grid, obs_type="ego",
                        ego_window=window)

    return _thunk


def evaluate(model, sizes, episodes: int, window: int,
             seed0: int = 1000) -> dict[int, float]:
    """Mean greedy score per board size (all episodes stepped as one batch)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    scores: dict[int, float] = {}
    for grid in sizes:
        vec = DummyVecEnv([make_env_fn(grid, window) for _ in range(episodes)])
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

def _load_demos(data_path: Path, device, cache: str):
    """Load the demo set, keeping the observations on the GPU when they fit.

    A BC step is a random gather of `batch` observations. At window 21 that is
    4.4 KB each, and shipping them host->device every step is a real cost; the
    whole set is only ~4.8 GB, so caching it in VRAM (when there is room for it
    *and* for training) makes the loop pure GPU. ``cache`` is
    ``auto``/``gpu``/``cpu``.
    """
    import torch

    data = np.load(data_path)
    obs = torch.from_numpy(data["obs"])           # (N,5,W,W) float16
    act = torch.from_numpy(data["act"])           # (N,) int64
    ret = torch.from_numpy(data["ret"])           # (N,) float32

    on_gpu = False
    if cache != "cpu" and device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Leave >= 45% of the free VRAM for parameters, activations and the
        # allocator's fragmentation headroom.
        fits = obs.nbytes + act.nbytes + ret.nbytes < free * 0.55
        if cache == "gpu" or fits:
            obs = obs.to(device)
            act = act.to(device)
            ret = ret.to(device)
            on_gpu = True
        else:
            print(f"[bc] demo set {obs.nbytes / 2**30:.1f} GiB does not fit in "
                  f"{free / 2**30:.1f} GiB free VRAM -> keeping it in host RAM",
                  flush=True)
    if not on_gpu and torch.cuda.is_available():
        obs = obs.pin_memory()  # pinned -> async, faster H2D copies
    return obs, act, ret, on_gpu


def behavior_clone(model, data_path: Path, epochs: int, batch_size: int,
                   lr: float, device, cache: str = "auto") -> None:
    import torch
    import torch.nn.functional as F

    obs_all, act_all, ret_all, on_gpu = _load_demos(data_path, device, cache)
    n = len(act_all)
    n_val = max(2048, n // 50)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    perm = perm.to(obs_all.device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[bc] {len(train_idx):,} train / {n_val:,} val transitions "
          f"(obs on {'GPU' if on_gpu else 'CPU'}, {obs_all.shape[1:]} )",
          flush=True)

    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, fused=(device.type == "cuda"))
    total_steps = epochs * (len(train_idx) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.05)

    def batch(idx):
        """Gather one minibatch as float32 on `device`."""
        o = obs_all[idx]
        if not on_gpu:
            o = o.to(device, non_blocking=True)
        return (o.to(torch.float32), act_all[idx].to(device, non_blocking=True),
                ret_all[idx].to(device, non_blocking=True))

    def mirror(obs_b, act_b):
        """Flip half of each batch left/right (ego frame): swap turn L/R."""
        m = torch.rand(len(act_b), device=obs_b.device) < 0.5
        obs_b = torch.where(m.view(-1, 1, 1, 1), obs_b.flip(-1), obs_b)
        swapped = torch.where(act_b == 1, torch.full_like(act_b, 2),
                              torch.where(act_b == 2, torch.full_like(act_b, 1), act_b))
        return obs_b, torch.where(m, swapped, act_b)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        tot_ce = tot_vf = tot_n = 0
        for i in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            obs_b, act_b, ret_b = batch(idx)
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
                obs_b, act_b, _ = batch(idx)
                dist = policy.get_distribution(obs_b)
                logits = dist.distribution.logits
                ce_sum += F.cross_entropy(logits, act_b, reduction="sum").item()
                correct += (logits.argmax(-1) == act_b).sum().item()
        policy.set_training_mode(True)
        print(f"[bc] epoch {epoch}/{epochs}: train ce={tot_ce / tot_n:.4f} "
              f"vf={tot_vf / tot_n:.3f} | val ce={ce_sum / n_val:.4f} "
              f"acc={correct / n_val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    policy.set_training_mode(False)
    del obs_all, act_all, ret_all, perm, train_idx, val_idx
    if on_gpu:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 3: PPO fine-tuning with a best-model gate
# ---------------------------------------------------------------------------

def ppo_finetune(model, timesteps: int, best_path: Path, eval_every: int,
                 window: int) -> None:
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
            scores = evaluate(self.model, self.gate_sizes, 5, window)
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
    parser.add_argument("--window", type=int, default=11,
                        help="ego observation window (odd, >=5). 11 = shipped "
                             "v2; 21 = wide view (4x the tokens)")
    parser.add_argument("--patch-size", type=int, default=1,
                        help="ViT patch size; tokens = (window/patch)^2")
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
    parser.add_argument("--bc-cache", choices=("auto", "gpu", "cpu"), default="auto",
                        help="keep the demo observations in VRAM (auto: if they fit)")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="bfloat16 autocast for the Transformer trunk on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=18)
    parser.add_argument("--ppo-batch", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()

    from gym_snake.obs import check_window

    window = check_window(args.window)
    suffix = "" if window == 11 else f"_w{window}"
    if args.out is None:
        args.out = REPO / "examples" / f"ppo_snake_transformer_v2{suffix}.zip"
    if args.work_dir is None:
        args.work_dir = Path(f"/tmp/snake_trf_v2{suffix}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------
    data_path = args.dataset
    if data_path is None or not data_path.exists():
        data_path = args.work_dir / "expert_demos.npz"
        if not data_path.exists():
            collect_dataset(args.transitions, args.workers, data_path, window,
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
    amp = args.amp and device == "cuda"
    print(f"[setup] device={device} window={window} patch={args.patch_size} "
          f"amp={amp}", flush=True)

    sizes = [MIX_SIZES[i % len(MIX_SIZES)] for i in range(args.n_envs)]
    vec_env = VecMonitor(SubprocVecEnv([make_env_fn(g, window) for g in sizes]))
    print(f"[setup] {args.n_envs} parallel envs, sizes={sizes}", flush=True)

    def linear_decay(progress_remaining: float) -> float:
        return 1e-4 * progress_remaining

    model = PPO(
        "CnnPolicy", vec_env, verbose=1, device=device,
        n_steps=1024, batch_size=args.ppo_batch, n_epochs=4,
        gamma=GAMMA, gae_lambda=0.95, ent_coef=0.005, clip_range=0.15,
        learning_rate=linear_decay,
        policy_kwargs=transformer_policy_kwargs(
            patch_size=args.patch_size, amp=amp, **MODEL_KWARGS),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    n_tokens = model.policy.features_extractor.n_tokens
    print(f"[setup] policy parameters: {n_params:,} ({n_tokens} tokens + CLS)",
          flush=True)

    # -- behavior cloning ---------------------------------------------------
    behavior_clone(model, data_path, args.bc_epochs, args.bc_batch,
                   args.bc_lr, model.device, args.bc_cache)
    bc_path = args.work_dir / "transformer_bc.zip"
    model.save(str(bc_path))
    print(f"[bc] saved BC model to {bc_path}", flush=True)

    bc_scores = evaluate(model, EVAL_SIZES, 10, window)
    print("[bc] eval:", {g: round(s, 2) for g, s in bc_scores.items()}, flush=True)

    if args.skip_ppo:
        model.save(str(args.out))
        vec_env.close()
        return

    # -- PPO fine-tune ------------------------------------------------------
    best_path = args.work_dir / "transformer_ppo_best.zip"
    ppo_finetune(model, args.timesteps, best_path, args.eval_every, window)
    vec_env.close()

    # -- final: pick best gate checkpoint, evaluate thoroughly --------------
    final = PPO.load(str(best_path if best_path.exists() else bc_path),
                     device=device)
    final_scores = evaluate(final, EVAL_SIZES, 20, window)
    print("[final] eval (20 eps/size):",
          {g: round(s, 2) for g, s in final_scores.items()}, flush=True)
    final.save(str(args.out))
    print(f"[final] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
