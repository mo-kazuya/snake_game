"""LoRA battle fine-tune of the ego/Transformer, merged to a drop-in checkpoint.

An alternative to the full fine-tune in ``train_transformer_battle.py``: instead
of updating every weight, this **freezes the shipped single-snake Transformer**
and learns only small **low-rank adapters** (LoRA) on its linear layers, via the
same battle behavior-cloning objective. The learned adapters are then **merged
back into the base weights** (``W <- W + (alpha/r)·B·A``) and saved as an
ordinary ``EgoTransformer`` PPO checkpoint — identical architecture to the base,
so the Django adapter loads it through the exact same ``PPO.load`` path with no
runtime changes (drop-in for the ``rl_trf_battle`` strategy).

Why LoRA here (and the honest caveats):

* **Pro** — the base single-snake capability is preserved exactly and only a
  small low-rank "battle delta" is trained, which limits forgetting and keeps
  the fine-tune disciplined.
* **Caveat** — this model is tiny (~0.88M params), so LoRA's usual memory /
  checkpoint-size wins don't matter, and it does **not** speed up CPU training:
  the bottleneck is the Transformer *forward* pass, which LoRA doesn't reduce.

Coverage: LoRA is injected into every ``nn.Linear`` we can target — the encoder
FFN (``linear1``/``linear2``), attention output projection (``out_proj``), the
feature head, and the SB3 policy/value MLPs and action/value heads. The packed
q/k/v projection inside ``nn.MultiheadAttention`` (``in_proj_weight``) is not an
``nn.Linear`` and is left frozen; the FFN + out_proj adapters carry the
adaptation. (This mirrors the common "attention-output + MLP" LoRA recipe.)

Run:

    # reuse a battle-demo dataset (e.g. from train_transformer_battle) and
    # produce a merged drop-in checkpoint
    python examples/train_transformer_lora.py --dataset /path/demos.npz

    # or collect demos first, then LoRA-train
    python examples/train_transformer_lora.py --bc-transitions 48000 --epochs 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Linear layers to adapt (matched by module-name suffix).
#
# Excluded on purpose: the whole ``self_attn`` block. Its q/k/v is a packed
# ``in_proj_weight`` (not an nn.Linear), and its ``out_proj``, although an
# nn.Linear, is consumed by ``nn.MultiheadAttention`` via the fused functional
# path that reads ``out_proj.weight``/``.bias`` as raw attributes rather than
# calling it as a module -- so replacing it with a wrapper breaks attention.
# The FFN (linear1/linear2) carries the bulk of each encoder block's linear
# params and is fully adaptable, so the adapters live there plus the heads.
TARGET_SUFFIXES = (
    "linear1", "linear2",                          # encoder FFN
    "features_extractor.head.0",                   # feature head
    "policy_net.0", "policy_net.2", "value_net.0", "value_net.2",  # SB3 MLPs
    "action_net", "value_net",                     # SB3 heads
)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def _make_lora_linear_cls():
    """Build the LoRALinear class lazily (needs torch imported)."""
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """Wrap a frozen ``nn.Linear`` with a trainable low-rank update.

        The effective weight is ``W + scaling·B·A`` (``A`` is ``(r, in)``,
        ``B`` is ``(out, r)``); ``B`` starts at zero so the adapter is a no-op
        at init.

        Crucially, the merged weight is exposed through **``weight``/``bias``
        properties**, not just ``forward``. ``nn.TransformerEncoderLayer`` (and
        ``nn.MultiheadAttention``) read ``linear.weight``/``.bias`` as raw
        attributes on their optimized path instead of calling the sub-linear as
        a module -- so a wrapper that only overrode ``forward`` would either
        crash (missing ``.weight``) or be silently bypassed. Returning a
        differentiable merged tensor from the property makes LoRA work through
        both the fused (eval) and the module-call (training) paths, and lets
        gradients reach ``A``/``B`` either way.
        """

        def __init__(self, base: nn.Linear, r: int, alpha: int):
            super().__init__()
            self.base = base
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)
            self.r = r
            self.scaling = alpha / r
            self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays 0

        @property
        def weight(self):
            return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            import torch.nn.functional as F

            return F.linear(x, self.weight, self.bias)

        @torch.no_grad()
        def merged_weight(self):
            return (self.base.weight + (self.lora_B @ self.lora_A) * self.scaling).detach()

    return LoRALinear


def inject_lora(policy, target_suffixes, r: int, alpha: int):
    """Replace targeted ``nn.Linear`` children of ``policy`` with LoRALinear.

    Returns ``(lora_modules, name_to_base_linear_name)`` where the mapping keys
    are the *base structure* weight names (``...linear1.weight``) used later to
    write merged weights into a clean checkpoint.
    """
    import torch.nn as nn

    LoRALinear = _make_lora_linear_cls()

    def matches(name: str) -> bool:
        return any(name == s or name.endswith("." + s) for s in target_suffixes)

    lora_modules: dict[str, object] = {}
    # Collect first (can't mutate while iterating named_modules).
    targets = [
        (name, mod) for name, mod in policy.named_modules()
        if isinstance(mod, nn.Linear) and matches(name)
    ]
    for name, mod in targets:
        parent = policy.get_submodule(name.rsplit(".", 1)[0]) if "." in name else policy
        attr = name.rsplit(".", 1)[-1]
        wrapped = LoRALinear(mod, r, alpha).to(mod.weight.device)
        setattr(parent, attr, wrapped)
        lora_modules[name] = wrapped
    return lora_modules


# ---------------------------------------------------------------------------
# Training (LoRA-only BC) and merge
# ---------------------------------------------------------------------------

def lora_bc(model, lora_modules, data_path: Path, epochs: int,
            batch_size: int, lr: float, device) -> None:
    import torch
    import torch.nn.functional as F

    data = np.load(data_path)
    obs_all = torch.from_numpy(data["obs"])
    act_all = torch.from_numpy(data["act"])
    ret_all = torch.from_numpy(data["ret"])
    n = len(act_all)
    n_val = max(2048, n // 50)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    policy = model.policy
    policy.set_training_mode(True)
    # Only LoRA params train.
    lora_params = [p for m in lora_modules.values() for p in (m.lora_A, m.lora_B)]
    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in policy.parameters())
    print(f"[lora] trainable adapters: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%) across {len(lora_modules)} layers",
          flush=True)
    opt = torch.optim.Adam(lora_params, lr=lr)
    total_steps = epochs * (len(train_idx) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.05)

    def mirror(obs_b, act_b):
        m = torch.rand(len(act_b), device=obs_b.device) < 0.5
        obs_b = torch.where(m.view(-1, 1, 1, 1), obs_b.flip(-1), obs_b)
        swapped = torch.where(act_b == 1, torch.full_like(act_b, 2),
                              torch.where(act_b == 2, torch.full_like(act_b, 1), act_b))
        return obs_b, torch.where(m, swapped, act_b)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(len(train_idx))]
        tot_ce = tot_n = 0
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
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            sched.step()
            tot_ce += ce.item() * len(idx)
            tot_n += len(idx)

        policy.set_training_mode(False)
        with torch.no_grad():
            correct = 0
            for i in range(0, n_val, batch_size):
                idx = val_idx[i:i + batch_size]
                obs_b = obs_all[idx].to(device, torch.float32)
                act_b = act_all[idx].to(device)
                logits = policy.get_distribution(obs_b).distribution.logits
                correct += (logits.argmax(-1) == act_b).sum().item()
        policy.set_training_mode(True)
        print(f"[lora] epoch {epoch}/{epochs}: train ce={tot_ce / tot_n:.4f} "
              f"| val acc={correct / n_val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    policy.set_training_mode(False)


def merge_to_clean_checkpoint(base_path: Path, lora_modules, device, out: Path):
    """Write LoRA-merged weights into a *fresh* base-structured checkpoint.

    The trained policy still has ``LoRALinear`` wrappers in it; rather than
    surgically unwrap them, we reload a clean base model (plain ``EgoTransformer``
    structure) and overwrite each targeted linear's weight with the merged
    ``W + scaling·B·A``. Everything else is unchanged (base was frozen), so the
    result is byte-compatible with the Django adapter's ``PPO.load``.
    """
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401

    clean = PPO.load(str(base_path), device=device)
    for name, lm in lora_modules.items():
        target = clean.policy.get_submodule(name)  # plain nn.Linear in the clean model
        target.weight.data.copy_(lm.merged_weight().to(target.weight.device))
    clean.save(str(out))
    return clean


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer.zip")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="battle-demo .npz (obs/act/ret); collected if absent")
    parser.add_argument("--bc-transitions", type=int, default=48_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA scaling alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path,
                        default=REPO / "examples" / "ppo_snake_transformer_battle.zip")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/snake_trf_lora"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------
    data_path = args.dataset
    if data_path is None or not data_path.exists():
        from examples.train_transformer_battle import collect_dataset

        data_path = args.work_dir / "battle_demos.npz"
        if not data_path.exists():
            collect_dataset(args.bc_transitions, args.workers, data_path, eps=0.1)

    # -- model + LoRA -------------------------------------------------------
    import torch
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, r={args.r}, alpha={args.alpha}", flush=True)

    model = PPO.load(str(args.init), device=device)
    lora_modules = inject_lora(model.policy, TARGET_SUFFIXES, args.r, args.alpha)

    # -- train LoRA + merge -------------------------------------------------
    lora_bc(model, lora_modules, data_path, args.epochs, args.batch, args.lr, device)
    merge_to_clean_checkpoint(args.init, lora_modules, device, args.out)
    print(f"[done] merged LoRA -> drop-in checkpoint at {args.out} "
          f"({args.out.stat().st_size // 1024} KiB)", flush=True)


if __name__ == "__main__":
    main()
