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

Coverage: LoRA is injected into the encoder FFN (``linear1``/``linear2``), the
feature head, the SB3 policy/value MLPs and action/value heads, **and the
self-attention** (on by default; ``--no-attn`` disables it). Attention needs a
**custom adapter** (:func:`inject_attn_lora` / ``AttnLoRA``) because its q/k/v is
a packed ``in_proj_weight`` (not an ``nn.Linear``) and its ``out_proj`` is
consumed through ``nn.MultiheadAttention``'s fused functional path -- neither is
reachable by wrapping a sub-``nn.Linear``. ``AttnLoRA`` freezes the attention
module and re-runs ``F.multi_head_attention_forward`` with LoRA-augmented q/k/v
and output projections instead. Plain ``nn.Linear`` layers use the simpler
``LoRALinear`` wrapper. Both merge into a clean checkpoint at the end.

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

# Plain ``nn.Linear`` layers to adapt with LoRALinear (matched by name suffix).
#
# The ``self_attn`` block is handled separately by :func:`inject_attn_lora`
# (AttnLoRA), NOT here: its q/k/v is a packed ``in_proj_weight`` (not an
# nn.Linear), and its ``out_proj`` is consumed by ``nn.MultiheadAttention`` via
# the fused functional path -- so neither can be adapted by wrapping a sub-Linear.
# What remains here is the encoder FFN (linear1/linear2), the feature head and
# the SB3 policy/value MLPs and heads.
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

        def lora_params(self):
            return [self.lora_A, self.lora_B]

        @torch.no_grad()
        def merge_into(self, target: "nn.Linear") -> None:
            """Write the merged weight into the matching plain ``nn.Linear`` of a
            clean (base-structured) checkpoint."""
            target.weight.data.copy_(self.merged_weight().to(target.weight.device))

    return LoRALinear


def _make_attn_lora_cls():
    """Build the AttnLoRA class lazily (wraps ``nn.MultiheadAttention``).

    ``nn.MultiheadAttention`` keeps its q/k/v as a **packed** ``in_proj_weight``
    parameter (not an ``nn.Linear``) and runs through ``F.multi_head_attention_forward``
    on the functional path, so it can't be adapted by wrapping a sub-``nn.Linear``.
    This wrapper instead **freezes the whole attention module** and adds low-rank
    adapters on both projections directly:

    * ``in_proj`` — one adapter on the packed ``(3·d, d)`` q/k/v matrix (covers q,
      k and v jointly),
    * ``out_proj`` — one adapter on the ``(d, d)`` output projection.

    ``forward`` re-runs ``F.multi_head_attention_forward`` with the LoRA-augmented
    weights (``W + scaling·B·A``), matching how ``nn.TransformerEncoderLayer``
    invokes self-attention (self-attn, ``need_weights=False``, ``batch_first``).
    Both ``B`` matrices start at zero so the adapter is a no-op at init.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttnLoRA(nn.Module):
        def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: int):
            super().__init__()
            self.mha = mha
            for p in self.mha.parameters():
                p.requires_grad_(False)
            d = mha.embed_dim
            self.r = r
            self.scaling = alpha / r
            self.in_A = nn.Parameter(torch.zeros(r, d))
            self.in_B = nn.Parameter(torch.zeros(3 * d, r))
            self.out_A = nn.Parameter(torch.zeros(r, d))
            self.out_B = nn.Parameter(torch.zeros(d, r))
            nn.init.kaiming_uniform_(self.in_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.out_A, a=math.sqrt(5))

        def __getattr__(self, name):
            # nn.TransformerEncoder / EncoderLayer read attributes straight off
            # ``self_attn`` (batch_first, num_heads, in_proj_weight, out_proj, ...)
            # when deciding their fast path. Delegate anything we don't define
            # ourselves to the wrapped module so the wrapper is a transparent
            # stand-in for nn.MultiheadAttention.
            try:
                return super().__getattr__(name)
            except AttributeError:
                mha = self._modules.get("mha")
                if mha is not None and hasattr(mha, name):
                    return getattr(mha, name)
                raise

        def eff_in_proj_weight(self):
            return self.mha.in_proj_weight + (self.in_B @ self.in_A) * self.scaling

        def eff_out_proj_weight(self):
            return self.mha.out_proj.weight + (self.out_B @ self.out_A) * self.scaling

        def forward(self, query, key, value, key_padding_mask=None,
                    need_weights=False, attn_mask=None, average_attn_weights=True,
                    is_causal=False):
            mha = self.mha
            is_batched = query.dim() == 3
            if mha.batch_first and is_batched:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
            attn_output, attn_weights = F.multi_head_attention_forward(
                query, key, value, mha.embed_dim, mha.num_heads,
                self.eff_in_proj_weight(), mha.in_proj_bias,
                mha.bias_k, mha.bias_v, mha.add_zero_attn,
                mha.dropout, self.eff_out_proj_weight(), mha.out_proj.bias,
                training=mha.training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal,
            )
            if mha.batch_first and is_batched:
                attn_output = attn_output.transpose(1, 0)
            return attn_output, attn_weights

        def lora_params(self):
            return [self.in_A, self.in_B, self.out_A, self.out_B]

        @torch.no_grad()
        def merge_into(self, target: nn.MultiheadAttention) -> None:
            target.in_proj_weight.data.copy_(
                self.eff_in_proj_weight().to(target.in_proj_weight.device))
            target.out_proj.weight.data.copy_(
                self.eff_out_proj_weight().to(target.out_proj.weight.device))

    return AttnLoRA


def inject_attn_lora(policy, r: int, alpha: int):
    """Wrap every ``nn.MultiheadAttention`` in ``policy`` with AttnLoRA.

    Returns a dict mapping the attention module's dotted name (e.g.
    ``features_extractor.encoder.layers.0.self_attn``) to its AttnLoRA, so
    :func:`merge_to_clean_checkpoint` can write the merged projections back into
    a clean checkpoint by the same name.
    """
    import torch.nn as nn

    AttnLoRA = _make_attn_lora_cls()

    targets = [
        (name, mod) for name, mod in policy.named_modules()
        if isinstance(mod, nn.MultiheadAttention)
    ]
    attn_modules: dict[str, object] = {}
    for name, mod in targets:
        parent = policy.get_submodule(name.rsplit(".", 1)[0]) if "." in name else policy
        attr = name.rsplit(".", 1)[-1]
        wrapped = AttnLoRA(mod, r, alpha).to(mod.in_proj_weight.device)
        setattr(parent, attr, wrapped)
        attn_modules[name] = wrapped
    return attn_modules


def inject_lora(policy, target_suffixes, r: int, alpha: int):
    """Replace targeted ``nn.Linear`` children of ``policy`` with LoRALinear.

    Returns ``lora_modules``: a dict mapping each wrapped layer's dotted name
    (e.g. ``features_extractor.encoder.layers.0.linear1``) to its LoRALinear.
    Those same names index plain ``nn.Linear`` layers in a fresh base model,
    which is how :func:`merge_to_clean_checkpoint` writes the merged weights
    back into a drop-in checkpoint.
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

def base_action_logits(model, data_path: Path, device, batch_size: int = 4096):
    """Greedy action logits of the *unadapted* policy over the whole dataset.

    Call before injecting LoRA. Returns ``(logits, mirrored_logits)`` on the
    CPU: the second is the base's answer to the horizontally flipped board, so
    the mirror augmentation in :func:`lora_bc` can look up the matching anchor
    instead of assuming the policy is exactly equivariant.
    """
    import torch

    obs_all = torch.from_numpy(np.load(data_path)["obs"])
    policy = model.policy
    policy.set_training_mode(False)
    out = []
    for flip in (False, True):
        chunks = []
        with torch.no_grad():
            for i in range(0, len(obs_all), batch_size):
                obs_b = obs_all[i:i + batch_size].to(device, torch.float32)
                if flip:
                    obs_b = obs_b.flip(-1)
                logits = policy.get_distribution(obs_b).distribution.logits
                chunks.append(logits.float().cpu())
        out.append(torch.cat(chunks))
    print(f"[anchor] base logits for {len(obs_all):,} states (+mirrored)", flush=True)
    return out[0], out[1]


def lora_bc(model, lora_modules, data_path: Path, epochs: int,
            batch_size: int, lr: float, device, anchor: float = 0.0,
            base_logits=None) -> None:
    """Behavior-clone the demos into the LoRA adapters.

    ``anchor`` > 0 adds ``anchor * KL(base || adapted)`` on every state, using
    the frozen base distributions from :func:`base_action_logits`. The adapter
    then only has to pay KL where the demonstrator actually disagrees with the
    base, so a *strong* base keeps its play everywhere else instead of being
    re-cloned down to the demonstrator's level.
    """
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
    # Only LoRA params train (works for both LoRALinear and AttnLoRA).
    lora_params = [p for m in lora_modules.values() for p in m.lora_params()]
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
        return obs_b, torch.where(m, swapped, act_b), m

    anchored = anchor > 0 and base_logits is not None
    if anchored:
        base_lg, base_lg_mir = base_logits
        print(f"[lora] anchoring to the base policy (KL weight {anchor})", flush=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(len(train_idx))]
        tot_ce = tot_kl = tot_n = 0
        for i in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            obs_b = obs_all[idx].to(device, torch.float32)
            act_b = act_all[idx].to(device)
            ret_b = ret_all[idx].to(device)
            obs_b, act_b, m = mirror(obs_b, act_b)

            if anchored:
                dist = policy.get_distribution(obs_b).distribution
                log_prob, entropy = dist.log_prob(act_b), dist.entropy()
                values = policy.predict_values(obs_b)
                # Anchor on the base's answer to the board the student sees:
                # the flipped copy for the mirrored half of the batch.
                ref = torch.where(m.view(-1, 1), base_lg_mir[idx].to(device),
                                  base_lg[idx].to(device))
                kl = F.kl_div(dist.logits.log_softmax(-1), ref.log_softmax(-1),
                              log_target=True, reduction="batchmean")
            else:
                values, log_prob, entropy = policy.evaluate_actions(obs_b, act_b)
                kl = torch.zeros((), device=device)
            ce = -log_prob.mean()
            vf = F.mse_loss(values.flatten(), ret_b)
            loss = ce + anchor * kl + 0.5 * vf - 0.003 * entropy.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            sched.step()
            tot_ce += ce.item() * len(idx)
            tot_kl += kl.item() * len(idx)
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
        kl_msg = f" kl={tot_kl / tot_n:.4f}" if anchored else ""
        print(f"[lora] epoch {epoch}/{epochs}: train ce={tot_ce / tot_n:.4f}{kl_msg} "
              f"| val acc={correct / n_val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    policy.set_training_mode(False)


def merge_to_clean_checkpoint(base_path: Path, lora_modules, device, out: Path):
    """Write LoRA-merged weights into a *fresh* base-structured checkpoint.

    The trained policy still has ``LoRALinear`` / ``AttnLoRA`` wrappers in it;
    rather than surgically unwrap them, we reload a clean base model (plain
    ``EgoTransformer`` structure) and let each wrapper write its merged
    ``W + scaling·B·A`` into the matching clean submodule (``merge_into``).
    Everything else is unchanged (base was frozen), so the result is
    byte-compatible with the Django adapter's ``PPO.load``.
    """
    from stable_baselines3 import PPO

    import gym_snake.policies  # noqa: F401

    clean = PPO.load(str(base_path), device=device)
    for name, lm in lora_modules.items():
        lm.merge_into(clean.policy.get_submodule(name))
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
    parser.add_argument("--no-attn", action="store_true",
                        help="skip the custom attention (qkv + out_proj) adapters "
                             "and adapt only the FFN + heads")
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
    if not args.no_attn:
        lora_modules.update(inject_attn_lora(model.policy, args.r, args.alpha))
        print(f"[setup] attention adapters on "
              f"{sum(1 for k in lora_modules if k.endswith('self_attn'))} MHA blocks",
              flush=True)

    # -- train LoRA + merge -------------------------------------------------
    lora_bc(model, lora_modules, data_path, args.epochs, args.batch, args.lr, device)
    merge_to_clean_checkpoint(args.init, lora_modules, device, args.out)
    print(f"[done] merged LoRA -> drop-in checkpoint at {args.out} "
          f"({args.out.stat().st_size // 1024} KiB)", flush=True)


if __name__ == "__main__":
    main()
