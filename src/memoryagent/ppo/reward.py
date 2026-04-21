"""Reward functions for the PPO stage.

REPLUG uses log-likelihood of the gold answer directly as its loss; it doesn't
need a separate reward. PPO does — the policy needs scalar feedback per
sampled trajectory. Two scalpel options today:

  - ExactMatchReward: 1.0 if the generated answer string-matches gold (after
    normalization), else 0.0. Brittle but unambiguous.
  - F1Reward: token-overlap F1. Smoother, partial credit. (Stub for now.)

Future: a learned reward model (RLHF-style preference model). Stub provided.
"""

from __future__ import annotations

import re
import string
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reward(Protocol):
    def __call__(self, question: str, doc: str, answer: str, gold: str | list[str]) -> float:
        ...


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


class ExactMatchReward:
    def __call__(self, question: str, doc: str, answer: str, gold: str | list[str]) -> float:
        candidates = [gold] if isinstance(gold, str) else gold
        ans_norm = _normalize(answer)
        return 1.0 if any(_normalize(g) == ans_norm for g in candidates) else 0.0


class F1Reward:
    def __call__(self, question: str, doc: str, answer: str, gold: str | list[str]) -> float:
        raise NotImplementedError("F1Reward stub — implement in PPO PR.")


class LearnedReward:
    """Placeholder for a future reward model fine-tuned on preference data."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def __call__(self, question: str, doc: str, answer: str, gold: str | list[str]) -> float:
        raise NotImplementedError("LearnedReward stub — wire to a real model in the PPO PR.")
