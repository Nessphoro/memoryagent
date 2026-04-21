"""Frozen reference generator for KL regularization (REPLUG +KL or PPO).

Mirrors QwenGenerator but loads with requires_grad=False and stays in eval
mode. Two consumers:
  1. REPLUG +KL regularizer (set `train.kl_to_ref > 0` in config) — keeps the
     trained generator from drifting too far from its starting point.
  2. PPO clipped objective (future) — needs a frozen reference for the ratio
     computation.

Right now nothing constructs this; it's plumbing for the PPO PR. Constructing
it is cheap if you have the model already on disk (no extra download).
"""

from __future__ import annotations

import torch

from memoryagent.models.generator import QwenGenerator


class FrozenReferenceGenerator:
    """Wraps a QwenGenerator, freezes it, exposes the same forward surface."""

    def __init__(self, gen: QwenGenerator):
        self.gen = gen
        self.gen.eval()
        for p in self.gen.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        *,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> FrozenReferenceGenerator:
        gen = QwenGenerator(
            model_name, dtype=dtype, attn_implementation=attn_implementation,
        )
        return cls(gen)

    @torch.no_grad()
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.gen.encode_query(input_ids, attention_mask)

    @torch.no_grad()
    def lm_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.gen.lm_forward(input_ids, attention_mask=attention_mask)

    def to(self, device) -> FrozenReferenceGenerator:
        self.gen.to(device)
        return self
