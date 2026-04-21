"""Muon helpers around ``torch.optim.Muon``.

PyTorch 2.11 ships Muon natively, so we just re-export it. What this module
adds is the surrounding plumbing torch's class doesn't ship:

  - ``partition_params_for_muon``: split ``(name, param)`` pairs into
    ``(muon, adamw)`` lists. Muon takes 2D inner weights; AdamW takes 1D
    params (norms, biases) and the embedding-like 2D matrices
    (``embed_tokens``, ``lm_head``, BERT ``embeddings.*``).
  - ``ChainedOptimizer``: a ``torch.optim.Optimizer``-shaped adapter over a
    list of optimizers, so ``LambdaLR`` and the rest of the train loop don't
    need to know there's a Muon+AdamW split underneath.

Use ``adjust_lr_fn="match_rms_adamw"`` (Moonshot's RMS-matching) on Muon so
the same per-module LR tuned for AdamW carries over without a multiplier.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Muon
from torch.optim.optimizer import Optimizer

__all__ = ["Muon", "ChainedOptimizer", "partition_params_for_muon"]


def partition_params_for_muon(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split ``(name, param)`` pairs into ``(muon_params, adamw_params)``.

    Muon: STRICTLY 2D params that are NOT embedding-like. ``torch.optim.Muon``
    rejects ndim != 2 at construction time, so 3D+ tensors (e.g. Qwen3.5's
    gated-deltanet conv weights) go to AdamW alongside biases and norms.

    AdamW: 1D params, 3D+ params, and embedding-like 2D matrices
    (``embed_tokens``, ``lm_head``, BERT ``embeddings.*``). Frozen params
    (``requires_grad=False``) are skipped entirely.
    """
    muon: list[torch.nn.Parameter] = []
    adam: list[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        is_embedding_like = any(
            tag in name for tag in ("embed_tokens", "lm_head", "embeddings", "word_embeddings")
        )
        if p.ndim == 2 and not is_embedding_like:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


class ChainedOptimizer(Optimizer):
    """Adapts a list of optimizers to the single-Optimizer interface.

    LambdaLR (and other LR schedulers) check ``isinstance(optim, Optimizer)``
    and iterate ``optim.param_groups`` to set per-group LRs. This wrapper:
      - inherits from ``Optimizer`` so the isinstance check passes
      - exposes a flat view of the underlying ``param_groups`` (the dicts are
        the SAME objects the children own — mutating ``g["lr"]`` writes
        through to the right child)
      - delegates ``step`` / ``zero_grad`` / ``state_dict`` to children

    We deliberately skip ``Optimizer.__init__`` (it requires a non-empty
    params list) and set the few attributes the LR scheduler reads.
    """

    def __init__(self, optimizers: list[Optimizer]):
        if not optimizers:
            raise ValueError("ChainedOptimizer needs at least one optimizer")
        self.optimizers = optimizers
        self.defaults = dict(optimizers[0].defaults)

    @property
    def param_groups(self) -> list[dict]:  # type: ignore[override]
        return [g for o in self.optimizers for g in o.param_groups]

    @param_groups.setter
    def param_groups(self, value):
        # Underlying optimizers manage their own param_groups; ignore writes.
        # (LR schedulers don't reassign param_groups, only mutate dict entries.)
        pass

    @property
    def state(self) -> dict:  # type: ignore[override]
        merged: dict = {}
        for o in self.optimizers:
            merged.update(o.state)
        return merged

    @state.setter
    def state(self, value):
        # Same rationale as param_groups: children own their state.
        pass

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def state_dict(self) -> dict:  # type: ignore[override]
        return {"chain": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd: dict) -> None:  # type: ignore[override]
        for o, child_sd in zip(self.optimizers, sd["chain"], strict=True):
            o.load_state_dict(child_sd)

    def add_param_group(self, param_group):  # type: ignore[override]
        raise NotImplementedError(
            "ChainedOptimizer doesn't support add_param_group; add to a child directly."
        )
