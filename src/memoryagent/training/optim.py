from __future__ import annotations

import torch

from memoryagent.config import OptimConfig
from memoryagent.device import DeviceCtx
from memoryagent.training.muon import (
    ChainedOptimizer,
    Muon,
    partition_params_for_muon,
)


def build_optimizer(
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    projection: torch.nn.Module,
    cfg: OptimConfig,
    ctx: DeviceCtx,
) -> torch.optim.Optimizer:
    """Build the optimizer chain matching ``cfg.optimizer``.

    AdamW path: a single ``torch.optim.AdamW`` (or ``AdamW8bit`` on CUDA when
    ``use_8bit_adam`` is set) with three param groups — one per module.

    Muon path: two optimizers chained behind a ``ChainedOptimizer``:
      - Muon over inner 2D matrices, with per-module LRs scaled up by
        ``muon_lr_multiplier``.
      - AdamW over 1D params (norms, biases) and embedding-like matrices
        (``embed_tokens``, ``lm_head``, BERT ``embeddings.*``), at the base
        per-module LRs.
    """
    if cfg.optimizer == "muon":
        return _build_muon_optimizer(generator, encoder, projection, cfg)
    return _build_adamw_optimizer(generator, encoder, projection, cfg, ctx)


def _build_adamw_optimizer(
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    projection: torch.nn.Module,
    cfg: OptimConfig,
    ctx: DeviceCtx,
) -> torch.optim.Optimizer:
    use_8bit = cfg.use_8bit_adam and ctx.use_8bit_adam
    if use_8bit:
        try:
            from bitsandbytes.optim import AdamW8bit as Adam
        except ImportError as e:
            raise RuntimeError(
                "use_8bit_adam=True but bitsandbytes is not installed. "
                "Install with `uv sync --extra cuda` (Linux+CUDA only)."
            ) from e
    else:
        Adam = torch.optim.AdamW

    return Adam(
        [
            {"params": list(generator.parameters()), "lr": cfg.lr_generator},
            {"params": list(encoder.parameters()), "lr": cfg.lr_encoder},
            {"params": list(projection.parameters()), "lr": cfg.lr_projection},
        ],
        weight_decay=cfg.weight_decay,
    )


def _build_muon_optimizer(
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    projection: torch.nn.Module,
    cfg: OptimConfig,
) -> torch.optim.Optimizer:
    gen_muon, gen_adam = partition_params_for_muon(generator.named_parameters())
    enc_muon, enc_adam = partition_params_for_muon(encoder.named_parameters())
    proj_muon, proj_adam = partition_params_for_muon(projection.named_parameters())

    # With adjust_lr_fn="match_rms_adamw" the per-module Muon LR is the same
    # number tuned for AdamW — Moonshot's RMS-matching does the rescaling
    # internally so the orthogonalized update has the same RMS as Adam's.
    muon_groups: list[dict] = []
    if gen_muon:
        muon_groups.append({"params": gen_muon, "lr": cfg.lr_generator})
    if enc_muon:
        muon_groups.append({"params": enc_muon, "lr": cfg.lr_encoder})
    if proj_muon:
        muon_groups.append({"params": proj_muon, "lr": cfg.lr_projection})
    if not muon_groups:
        raise RuntimeError("Muon path selected but no matrix params found.")
    muon = Muon(
        muon_groups,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.muon_ns_steps,
        weight_decay=cfg.weight_decay,
        adjust_lr_fn=cfg.muon_adjust_lr_fn,
    )

    adam_groups: list[dict] = []
    if gen_adam:
        adam_groups.append({"params": gen_adam, "lr": cfg.lr_generator})
    if enc_adam:
        adam_groups.append({"params": enc_adam, "lr": cfg.lr_encoder})
    if proj_adam:
        adam_groups.append({"params": proj_adam, "lr": cfg.lr_projection})
    if not adam_groups:
        # All params went to Muon — unlikely but possible. Return Muon alone.
        return muon
    adam = torch.optim.AdamW(adam_groups, weight_decay=cfg.weight_decay)

    return ChainedOptimizer([muon, adam])


def build_scheduler(
    optim: torch.optim.Optimizer,
    cfg: OptimConfig,
    *,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then constant. Keep simple — research code, not production."""
    warmup = max(cfg.warmup_steps, 1)
    return torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup),
    )
