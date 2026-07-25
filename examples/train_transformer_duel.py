"""Fine-tune the ego/Transformer into a two-snake *battle* specialist.

Starts from the shipped single-snake Transformer (``ppo_snake_transformer.zip``)
and continues PPO training inside :class:`examples.duel_env.SnakeDuelEnv`, a
two-snake board that reuses the deployed engine's rules and the deployed
egocentric observation (opponent folded into the deadly/body channels). The
result is saved as a *new* model, ``ppo_snake_transformer_duel.zip``, and
registered in the Django app as the ``rl_trf_duel`` strategy -- the shipped
single-snake models are left untouched.

Opponent mix (self-play + a strong scripted baseline)
-----------------------------------------------------
Half the parallel envs face the **search AI** (``game.ai`` -- BFS/flood-fill,
already opponent-aware) and half face a **frozen snapshot of the learner**
that is refreshed to the latest weights every ``--refresh`` steps. Facing both
a fixed strong opponent and an evolving copy of itself keeps training stable
while still producing a policy that holds up against neural opponents.

Run (GPU + many CPU cores):

    python examples/train_transformer_duel.py                 # full run
    python examples/train_transformer_duel.py --timesteps 4000000 --n-envs 16

Evaluation is head-to-head: the candidate (snake 0, greedy) vs an opponent
(snake 1) over full games, reporting mean score and win rate per board size.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

# Two snakes need a little room, so skip the very smallest boards.
MIX_SIZES = [10, 12, 14, 16, 20, 24, 30, 40]
EVAL_SIZES = [12, 16, 20, 30]

GAMMA = 0.99
# Same architecture as the shipped Transformer so we can warm-start its weights.
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4,
                    dim_feedforward=512, patch_size=1)

_HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_DIR_NAMES = ["up", "right", "down", "left"]


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def make_env_fn(grid: int, opponent: str, opp_path: str | None):
    def _thunk():
        # Re-add paths inside the subprocess worker.
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "examples"))
        import gym_snake  # noqa: F401
        from duel_env import SnakeDuelEnv

        return SnakeDuelEnv(grid_size=grid, opponent=opponent,
                            opponent_model_path=opp_path)

    return _thunk


# ---------------------------------------------------------------------------
# Head-to-head evaluation (full games via the real engine)
# ---------------------------------------------------------------------------

def _model_dir(model, state, snake_index):
    from game.engine import DIRECTIONS
    from gym_snake.obs import ego_observation

    me = state.snakes[snake_index]
    opp = [c for j, o in enumerate(state.snakes)
           if j != snake_index and o.alive for c in o.body]
    obs = ego_observation(
        me.body, state.food, state.grid,
        _HEADINGS.index(DIRECTIONS[me.direction]), opponent_cells=opp,
    )
    action, _ = model.predict(obs, deterministic=True)
    hidx = _HEADINGS.index(DIRECTIONS[me.direction])
    a = int(action)
    if a == 1:
        hidx = (hidx + 1) % 4
    elif a == 2:
        hidx = (hidx - 1) % 4
    return _DIR_NAMES[hidx]


def _search_dir(state, snake_index):
    from game.ai import choose_direction
    return choose_direction(state, snake_index)


def duel_eval(candidate, opponent_fn, sizes, games: int, seed0: int = 7000):
    """Candidate (snake 0, greedy) vs opponent (snake 1) over full games.

    ``opponent_fn(state) -> direction`` drives snake 1. Returns per size a
    dict with mean candidate score, mean opponent score and candidate win rate
    (win = strictly higher score when the game ends)."""
    import random

    from game.engine import GameState

    out = {}
    for grid in sizes:
        cand_scores, opp_scores, wins = [], [], 0
        for g in range(games):
            random.seed(seed0 + g)
            state = GameState(grid=grid)
            state.reset(["cand", "opp"])
            guard = grid * grid * 4
            while not state.game_over and guard > 0:
                guard -= 1
                dirs = []
                dirs.append(_model_dir(candidate, state, 0)
                            if state.snakes[0].alive else state.snakes[0].direction)
                dirs.append(opponent_fn(state, 1)
                            if state.snakes[1].alive else state.snakes[1].direction)
                state.step(dirs)
            cs, os_ = state.snakes[0].score, state.snakes[1].score
            cand_scores.append(cs)
            opp_scores.append(os_)
            wins += 1 if cs > os_ else 0
        out[grid] = {
            "cand": float(np.mean(cand_scores)),
            "opp": float(np.mean(opp_scores)),
            "win": wins / games,
        }
    return out


def summarize(results) -> tuple[float, float, str]:
    """Mean candidate score and mean win rate across sizes, plus a log line."""
    cand = float(np.mean([r["cand"] for r in results.values()]))
    win = float(np.mean([r["win"] for r in results.values()]))
    line = " ".join(f"{g}:{r['cand']:.0f}/{r['opp']:.0f}(w{r['win']:.0%})"
                    for g, r in results.items())
    return cand, win, line


# ---------------------------------------------------------------------------
# Self-play opponent refresh
# ---------------------------------------------------------------------------

def make_refresh_callback(vec_env, snapshot_path: Path, refresh_every: int,
                          eval_every: int, best_path: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class DuelCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_refresh = refresh_every
            self.next_eval = eval_every
            self.best = -np.inf

        def _refresh_opponent(self):
            self.model.save(str(snapshot_path))
            # Broadcast to every worker; search-opponent envs ignore it.
            vec_env.env_method("refresh_opponent", str(snapshot_path))
            print(f"[selfplay] refreshed opponent snapshot at "
                  f"{self.num_timesteps:,} steps", flush=True)

        def _eval_and_gate(self):
            res = duel_eval(self.model, _search_dir, (12, 20, 30), games=6)
            cand, win, line = summarize(res)
            metric = cand + 40.0 * win   # score plus a strong win-rate bonus
            marker = ""
            if metric > self.best:
                self.best = metric
                self.model.save(str(best_path))
                marker = "  <- new best, saved"
            print(f"[gate] steps={self.num_timesteps:,} vs-search "
                  f"cand={cand:.1f} win={win:.0%} [{line}] metric={metric:.1f}{marker}",
                  flush=True)

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_refresh:
                self.next_refresh += refresh_every
                self._refresh_opponent()
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                self._eval_and_gate()
            return True

        def _on_training_end(self) -> None:
            self._eval_and_gate()

    return DuelCallback()


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer.zip")
    parser.add_argument("--timesteps", type=int, default=6_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--refresh", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer_duel.zip")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_duel"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from gym_snake.policies import transformer_policy_kwargs

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    if not args.base.exists():
        raise SystemExit(f"base model not found: {args.base}")

    # The self-play opponent is bootstrapped from the base model, then
    # refreshed to newer learner snapshots during training.
    snapshot_path = args.work_dir / "opponent.zip"
    import shutil
    shutil.copy(args.base, snapshot_path)

    # Half the envs face the search AI, half face the (evolving) self snapshot.
    env_fns = []
    for i in range(args.n_envs):
        grid = MIX_SIZES[i % len(MIX_SIZES)]
        if i % 2 == 0:
            env_fns.append(make_env_fn(grid, "search", None))
        else:
            env_fns.append(make_env_fn(grid, "model", str(snapshot_path)))
    vec_env = VecMonitor(SubprocVecEnv(env_fns))
    print(f"[setup] {args.n_envs} envs "
          f"({args.n_envs // 2} vs-search, {args.n_envs - args.n_envs // 2} self-play)",
          flush=True)

    def lr_decay(progress_remaining: float) -> float:
        return args.lr * progress_remaining

    model = PPO(
        "CnnPolicy", vec_env, verbose=0, device=device,
        n_steps=1024, batch_size=1024, n_epochs=4,
        gamma=GAMMA, gae_lambda=0.95, ent_coef=args.ent_coef, clip_range=0.15,
        learning_rate=lr_decay,
        policy_kwargs=transformer_policy_kwargs(**MODEL_KWARGS),
    )

    # Warm-start every weight from the shipped Transformer.
    base = PPO.load(str(args.base), device=device)
    model.policy.load_state_dict(base.policy.state_dict())
    del base
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[setup] warm-started {n_params:,} params from {args.base.name}", flush=True)

    # Baseline: how does the un-fine-tuned model already do head-to-head?
    print("[baseline] evaluating warm-started model vs search AI...", flush=True)
    base_res = duel_eval(model, _search_dir, (12, 20, 30), games=6)
    cand, win, line = summarize(base_res)
    print(f"[baseline] vs-search cand={cand:.1f} win={win:.0%} [{line}]", flush=True)

    best_path = args.work_dir / "duel_best.zip"
    cb = make_refresh_callback(vec_env, snapshot_path, args.refresh,
                               args.eval_every, best_path)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb,
                reset_num_timesteps=True, progress_bar=False)
    print(f"[done] training took {(time.time() - t0) / 60:.1f} min", flush=True)
    vec_env.close()

    # Final: reload the best gated checkpoint and evaluate thoroughly, both
    # against the search AI and against the original single-snake Transformer.
    final = PPO.load(str(best_path if best_path.exists() else args.out), device=device)

    print("[final] vs search AI:", flush=True)
    res = duel_eval(final, _search_dir, EVAL_SIZES, games=12)
    cand, win, line = summarize(res)
    print(f"   cand={cand:.1f} win={win:.0%} [{line}]", flush=True)

    import gym_snake.policies  # noqa: F401
    orig = PPO.load(str(args.base), device="cpu")

    def orig_dir(state, i):
        return _model_dir(orig, state, i)

    print("[final] vs original single-snake Transformer:", flush=True)
    res2 = duel_eval(final, orig_dir, EVAL_SIZES, games=12)
    cand2, win2, line2 = summarize(res2)
    print(f"   cand={cand2:.1f} win={win2:.0%} [{line2}]", flush=True)

    final.save(str(args.out))
    print(f"[final] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
