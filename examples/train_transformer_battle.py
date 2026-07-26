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
   observation space ``(5, W, W)`` and action space ``Discrete(3)`` are
   identical to the battle env, so the weights transfer directly — no surgery.
   (Use ``--no-init`` to train the same architecture from scratch instead.)
   ``--window`` selects the ego window (see ``gym_snake/obs.py``); it must match
   the ``--init`` model's, and the defaults for ``--init``/``--out`` follow it,
   so ``--window 21`` warm-starts the 21x21 model into a 21x21 battle model.
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

    # full GPU recipe: opponent-aware BC warm start, then 8M PPO steps
    python examples/train_transformer_battle.py --bc-transitions 600000

    # ...on top of the wide-view (21x21) single-snake model
    python examples/train_transformer_battle.py --window 21 --bc-transitions 600000

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

# The terminal win/lose bonus is what makes this a *battle* objective, so the
# discount has to reach it. At 0.99 a win 300 ticks away is worth 0.05 -- less
# than a twentieth of a single food -- and 40x40 games run 300-600 ticks, so the
# old runs optimized eating and only incidentally winning (food went up 6.6x,
# win rate only 32%->47%). 0.997 gives a ~330-tick horizon, which actually
# covers a game.
GAMMA = 0.997

# The value target scales like 1/(1-gamma), so 0.99 -> 0.997 makes returns ~3.3x
# bigger and the *squared* error ~11x bigger. The policy and value heads share
# the Transformer trunk, so leaving the usual 0.5 here would let value fitting
# drive the shared representation (measured: BC val accuracy fell 90.6% -> 87.7%
# on the same demos purely from raising gamma). Scaling the coefficient down by
# the same ~11x restores the balance the 0.99 runs had.
VF_COEF = 0.05

# Same architecture as the shipped v2 Transformer, so --init weights load
# straight in (and --no-init reproduces that architecture from scratch).
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4, dim_feedforward=512)

# Battle-specific reward bonuses layered on top of the SnakeEnv-identical base.
REWARD_OPP_DEATH = 0.5   # once, when the opponent dies while we're alive
REWARD_WIN = 1.0         # terminal, if our score > opponent's
REWARD_LOSE = -1.0       # terminal, if our score < opponent's

# Playstyle *characters*, expressed as reward profiles rather than as imitation
# of a styled expert. Cloning a search-AI-based teacher stops working once the
# base outclasses it: measured against the search AI, the battle base kills the
# rival in 83% of games while an aggression-cloned LoRA on top of it manages
# 23%, i.e. the "aggressive" character came out *less* aggressive and weaker on
# every axis. A reward says what the character wants without capping it at a
# teacher's skill, so the character can be **stronger** than the base at its own
# speciality.
# Sizing matters more than sign here. A good policy eats 40+ apples per episode,
# so the food term alone is worth +40; a 3.0 kill bonus or a 0.01/tick survival
# wage is 7-12% of that and vanishes into the noise. Measured: characters
# trained that way were behaviourally identical to the neutral policy (and to
# each other) on every axis -- distance, kills, survival. The style term has to
# be able to *outbid* a few apples before it changes what the policy does.
STYLE_REWARDS = {
    # The battle objective itself: the neutral middle of the three.
    "balanced": dict(opp_death=REWARD_OPP_DEATH, alive=0.0,
                     win=REWARD_WIN, lose=REWARD_LOSE),
    # A kill is worth ~30 apples, so hunting beats grazing.
    "aggressive": dict(opp_death=30.0, alive=0.0,
                       win=REWARD_WIN, lose=REWARD_LOSE),
    # Pure survival: no bounty on the rival, no interest in the final score,
    # and a wage of 0.15/tick -- over a ~500-tick game that is worth more than
    # everything it could have eaten.
    "defensive": dict(opp_death=0.0, alive=0.15, win=0.0, lose=0.0),
}

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# ---------------------------------------------------------------------------
# Phase 1: expert *battle* demonstrations (search AI vs search AI, parallel)
# ---------------------------------------------------------------------------

def _collect_chunk(args: tuple[int, int, int, float, int, bool]):
    """Play the search AI as snake 0 against a search-AI opponent on `grid`
    boards until `quota` transitions for snake 0 exist.

    The recorded observation is snake 0's egocentric view **with the live
    opponent body folded in** (``opponent_cells``), the label is the expert's
    relative action, and the return uses the battle env's base reward scheme
    (used to warm-start the value head). With probability `eps` a random *safe*
    move is executed instead of the expert's (DAgger-style off-distribution
    coverage) but the label stays the expert's.
    """
    seed, grid, quota, eps, window, opp_channels = args
    random.seed(seed)

    from game.ai import _opposite, _simulate, choose_direction
    from game.engine import DIRECTIONS, GameState
    from gym_snake.envs.battle_env import _heading_index
    from gym_snake.obs import ego_observation

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

        while not state.game_over and state.snakes[0].alive and steps_since_food < grid * grid:
            agent = state.snakes[0]
            hidx = _heading_index(agent.direction)
            opp = rivals(state, 0)
            obs = ego_observation(agent.body, state.food, grid, hidx,
                                  window=window,
                                  opponent_cells=[c for s_ in opp for c in s_.body],
                                  opponent_head=opp[0].body[0] if opp else None,
                                  opponent_channels=opp_channels)

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


def collect_dataset(total: int, workers: int, out_path: Path, window: int,
                    opp_channels: bool = False, eps: float = 0.1) -> Path:
    per_size = total // len(MIX_SIZES)
    chunk = 20_000
    tasks = []
    seed = 0
    for grid in MIX_SIZES:
        remaining = per_size
        while remaining > 0:
            q = min(chunk, remaining)
            tasks.append((seed, grid, q, eps, window, opp_channels))
            seed += 1
            remaining -= q
    random.shuffle(tasks)

    print(f"[collect] {len(tasks)} chunks on {workers} workers "
          f"({per_size:,} transitions x {len(MIX_SIZES)} sizes, window={window}, "
          f"channels={8 if opp_channels else 5})",
          flush=True)
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

def make_env_fn(grid: int, opponent: str, window: int,
                opp_channels: bool = False, style: str = "balanced"):
    profile = STYLE_REWARDS[style]

    def _thunk():
        import gymnasium as gym

        import gym_snake  # noqa: F401

        return gym.make(
            "gym_snake/SnakeBattle-v0", grid_size=grid, opponent=opponent,
            ego_window=window, reward_opp_death=profile["opp_death"],
            reward_win=profile["win"], reward_lose=profile["lose"],
            reward_alive=profile["alive"], opponent_channels=opp_channels,
        )

    return _thunk


def evaluate(model, sizes, episodes: int, window: int,
             opponent: str = "search", opp_channels: bool = False,
             style: str = "balanced", seed0: int = 1000):
    """Greedy eval vs a fixed opponent.

    Returns ``(mean food, win-rate, mean episode return)`` per size. The return
    is measured under ``style``'s reward, which is what a *character* is
    actually trying to maximize -- gating on food alone would just pick the
    checkpoint that drifted back towards the neutral policy.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    food: dict[int, float] = {}
    winrate: dict[int, float] = {}
    ep_return: dict[int, float] = {}
    for grid in sizes:
        vec = DummyVecEnv([make_env_fn(grid, opponent, window, opp_channels, style)
                           for _ in range(episodes)])
        vec.seed(seed0)
        obs = vec.reset()
        finished = np.zeros(episodes, dtype=bool)
        eps_food = np.zeros(episodes)
        eps_win = np.zeros(episodes)
        eps_ret = np.zeros(episodes)
        running = np.zeros(episodes)
        while not finished.all():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec.step(actions)
            running += np.asarray(rewards) * ~finished
            for i in range(episodes):
                if dones[i] and not finished[i]:
                    finished[i] = True
                    eps_food[i] = infos[i]["score"]
                    eps_win[i] = 1.0 if infos[i].get("win") else 0.0
                    eps_ret[i] = running[i]
        vec.close()
        food[grid] = float(eps_food.mean())
        winrate[grid] = float(eps_win.mean())
        ep_return[grid] = float(eps_ret.mean())
    return food, winrate, ep_return


# ---------------------------------------------------------------------------
# Phase 2: behavior cloning (identical recipe to train_transformer_v2)
# ---------------------------------------------------------------------------

def _load_demos(data_path: Path, device, cache: str):
    """Load the battle demos, keeping the observations in VRAM when they fit.

    Same trick as ``train_transformer_v2._load_demos``: a BC step is a random
    gather, so caching the whole (float16) set on the GPU removes the
    host->device copy from every step. ``cache`` is ``auto``/``gpu``/``cpu``.
    """
    import torch

    data = np.load(data_path)
    obs = torch.from_numpy(data["obs"])           # (N,5,W,W) float16
    act = torch.from_numpy(data["act"])           # (N,) int64
    ret = torch.from_numpy(data["ret"])           # (N,) float32

    on_gpu = False
    if cache != "cpu" and device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if cache == "gpu" or obs.nbytes + act.nbytes + ret.nbytes < free * 0.55:
            obs, act, ret = obs.to(device), act.to(device), ret.to(device)
            on_gpu = True
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
          f"(obs on {'GPU' if on_gpu else 'CPU'})", flush=True)

    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr,
                           fused=(device.type == "cuda"))
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
            loss = ce + VF_COEF * vf - 0.003 * entropy.mean()

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
# Phase 3: PPO fine-tuning with a best-by-battle-eval gate
# ---------------------------------------------------------------------------

def ppo_finetune(model, timesteps: int, best_path: Path, eval_every: int,
                 window: int, opp_channels: bool = False,
                 gate_episodes: int = 20, style: str = "balanced",
                 gate_metric: str = "combined") -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class BestGate(BaseCallback):
        """Every `eval_every` steps evaluate vs the search AI on small/mid/large
        boards and keep the checkpoint with the best combined score (mean food
        plus a win-rate bonus), so the gate rewards *winning*, not just eating.

        The episode count matters more than it looks: at 5 episodes per size an
        earlier run scored 25.9 food / 60% win and 40.5 food / 47% win **18k
        steps apart**, i.e. the gate was mostly selecting noise. 20 episodes on
        fixed seeds costs a couple of minutes per gate and actually ranks
        checkpoints."""

        gate_sizes = (10, 20, 40)

        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.next_eval = eval_every

        def _eval_and_save(self):
            food, winrate, rets = evaluate(
                self.model, self.gate_sizes, gate_episodes, window,
                opp_channels=opp_channels, style=style)
            mean_food = float(np.mean(list(food.values())))
            mean_win = float(np.mean(list(winrate.values())))
            mean_ret = float(np.mean(list(rets.values())))
            combined = (mean_ret if gate_metric == "return"
                        else mean_food + 10.0 * mean_win)
            line = " ".join(f"{g}:{food[g]:.1f}/{winrate[g]:.0%}" for g in self.gate_sizes)
            marker = ""
            if combined > self.best:
                self.best = combined
                self.model.save(str(best_path))
                marker = "  <- new best, saved"
            print(f"[gate] steps={self.num_timesteps:,} food={mean_food:.2f} "
                  f"win={mean_win:.0%} ret={mean_ret:.1f} ({line}){marker}",
                  flush=True)

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

def build_model(vec_env, init_path: Path | None, device,
                patch_size: int = 1, amp: bool = False):
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
        vf_coef=VF_COEF,
        learning_rate=lambda pr: 1e-4 * pr,
        policy_kwargs=transformer_policy_kwargs(
            patch_size=patch_size, amp=amp, **MODEL_KWARGS),
    )

    if init_path is not None and init_path.exists():
        print(f"[setup] warm-starting policy weights from {init_path}", flush=True)
        source = PPO.load(str(init_path), device=device)
        src_shape = tuple(source.observation_space.shape)
        dst_shape = tuple(vec_env.observation_space.shape)
        if src_shape[1:] != dst_shape[1:] or src_shape[0] > dst_shape[0]:
            raise SystemExit(
                f"--init model was trained with observation {src_shape} but "
                f"this run uses {dst_shape}: pass the matching --window (the "
                f"ego window is part of the weights)."
            )
        state_dict = source.policy.state_dict()
        if src_shape[0] < dst_shape[0]:
            # Extra observation channels: keep the base's patch embedding for
            # the channels it knows and zero-init the new ones, so the policy
            # starts out *numerically identical* to --init and then learns what
            # the added channels are worth. (Only the patch embedding reads the
            # observation directly, so nothing else needs resizing.)
            import torch

            target = model.policy.state_dict()
            grown = []
            for key, src_w in list(state_dict.items()):
                dst_w = target.get(key)
                if dst_w is None or dst_w.shape == src_w.shape:
                    continue
                if not key.endswith("embed.weight"):
                    raise SystemExit(
                        f"cannot grow {key} from {tuple(src_w.shape)} to "
                        f"{tuple(dst_w.shape)}: only the patch embedding is "
                        f"expected to depend on the channel count."
                    )
                w = torch.zeros_like(dst_w)
                w[:, :src_w.shape[1]] = src_w
                state_dict[key] = w
                grown.append(key)
            print(f"[setup] grew {src_shape[0]} -> {dst_shape[0]} observation "
                  f"channels; zero-init on the new ones in {len(grown)} "
                  f"embedding(s)", flush=True)
        missing, unexpected = model.policy.load_state_dict(
            state_dict, strict=False)
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
    parser.add_argument("--window", type=int, default=11,
                        help="ego observation window (odd, >=5). Must match the "
                             "--init model's window")
    parser.add_argument("--patch-size", type=int, default=1,
                        help="ViT patch size; tokens = (window/patch)^2")
    parser.add_argument("--init", type=Path, default=None,
                        help="single-snake Transformer to warm-start from "
                             "(default: the shipped model for --window)")
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
    parser.add_argument("--bc-cache", choices=("auto", "gpu", "cpu"), default="auto",
                        help="keep the demo observations in VRAM (auto: if they fit)")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="bfloat16 autocast for the Transformer trunk on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--style", choices=tuple(STYLE_REWARDS), default="balanced",
                        help="reward profile / playstyle character to train")
    parser.add_argument("--gate-metric", choices=("combined", "return"),
                        default="combined",
                        help="what the gate ranks checkpoints by: 'combined' "
                             "(mean food + 10x win rate) or 'return' (mean "
                             "episode return under --style's own reward, which "
                             "is what a character is optimizing)")
    parser.add_argument("--gate-episodes", type=int, default=20,
                        help="episodes per board size in the gate eval; too few "
                             "and the gate selects noise instead of skill")
    parser.add_argument("--opponent-channels", dest="opp_channels",
                        action="store_true", default=True,
                        help="give the policy the 3 opponent-aware observation "
                             "channels (rival body / rival head, local + minimap)")
    parser.add_argument("--no-opponent-channels", dest="opp_channels",
                        action="store_false")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()

    from gym_snake.obs import check_window

    window = check_window(args.window)
    suffix = "" if window == 11 else f"_w{window}"
    opp_channels = args.opp_channels
    short = {"balanced": "bal", "aggressive": "aggr", "defensive": "def"}
    styled = args.style != "balanced" or args.gate_metric == "return"
    if args.init is None:
        # A character starts from the finished battle model; the plain recipe
        # starts from the single-snake one.
        args.init = REPO / "examples" / (
            f"ppo_snake_transformer_battle{suffix}.zip" if styled
            else f"ppo_snake_transformer{suffix}.zip")
    if args.out is None:
        args.out = REPO / "examples" / (
            f"ppo_snake_transformer_battle{suffix}_{short[args.style]}.zip"
            if styled else f"ppo_snake_transformer_battle{suffix}.zip")
    if args.work_dir is None:
        args.work_dir = Path(f"/tmp/snake_trf_battle{suffix}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    init_path = None if args.no_init else args.init

    # -- optional battle-demo dataset ---------------------------------------
    data_path = args.dataset
    if args.bc_transitions > 0 and data_path is None:
        # The demos carry the observation, so channel settings can't share a
        # cache file.
        data_path = args.work_dir / (
            "battle_demos_opp.npz" if opp_channels else "battle_demos.npz")
        if not data_path.exists():
            collect_dataset(args.bc_transitions, args.workers, data_path,
                            window, opp_channels, eps=args.demo_eps)
        else:
            print(f"[collect] reusing cached {data_path}", flush=True)

    # -- model + envs -------------------------------------------------------
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"
    print(f"[setup] device={device} window={window} patch={args.patch_size} "
          f"amp={amp} channels={8 if opp_channels else 5} gamma={GAMMA} "
          f"style={args.style} reward={STYLE_REWARDS[args.style]}", flush=True)

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    sizes = [MIX_SIZES[i % len(MIX_SIZES)] for i in range(args.n_envs)]
    env_fns = [make_env_fn(sizes[i], opponents[i % len(opponents)], window,
                          opp_channels, args.style)
               for i in range(args.n_envs)]
    vec_env = VecMonitor(SubprocVecEnv(env_fns))
    print(f"[setup] {args.n_envs} envs, sizes={sizes}, opponents={opponents}", flush=True)

    model = build_model(vec_env, init_path, device, args.patch_size, amp)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[setup] policy parameters: {n_params:,}", flush=True)

    # -- optional behavior cloning ------------------------------------------
    if data_path is not None:
        behavior_clone(model, data_path, args.bc_epochs, args.bc_batch,
                       args.bc_lr, model.device, args.bc_cache)
        bc_path = args.work_dir / "battle_bc.zip"
        model.save(str(bc_path))
        food, win, _ = evaluate(model, EVAL_SIZES, 10, window,
                                opp_channels=opp_channels, style=args.style)
        print("[bc] eval food:", {g: round(v, 2) for g, v in food.items()},
              "win:", {g: round(v, 2) for g, v in win.items()}, flush=True)

    if args.skip_ppo:
        model.save(str(args.out))
        vec_env.close()
        print(f"[done] saved (no PPO) to {args.out}", flush=True)
        return

    # -- PPO fine-tune ------------------------------------------------------
    best_path = args.work_dir / "battle_ppo_best.zip"
    ppo_finetune(model, args.timesteps, best_path, args.eval_every, window,
                 opp_channels=opp_channels, gate_episodes=args.gate_episodes,
                 style=args.style, gate_metric=args.gate_metric)
    vec_env.close()

    # -- final: pick best gate checkpoint, evaluate thoroughly --------------
    from stable_baselines3 import PPO

    final = PPO.load(str(best_path if best_path.exists() else args.out
                         if args.out.exists() else best_path), device=device)
    food, win, rets = evaluate(final, EVAL_SIZES, 20, window,
                               opp_channels=opp_channels, style=args.style)
    print("[final] food (20 eps/size):", {g: round(v, 2) for g, v in food.items()}, flush=True)
    print("[final] win-rate vs search:", {g: round(v, 2) for g, v in win.items()}, flush=True)
    print("[final] episode return:", {g: round(v, 1) for g, v in rets.items()}, flush=True)
    final.save(str(args.out))
    print(f"[final] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
