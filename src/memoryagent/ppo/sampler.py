"""Retrieval samplers for the REPLUG / PPO split.

REPLUG (today) uses ``TopKRetrieval``: deterministic top-k from the score
softmax. PPO (future) will use ``SampledRetrieval``: sample k docs without
replacement and store the sampling log-probs for the policy gradient.

Both implement the same protocol so ``replug_step`` and the eventual
``ppo_step`` can swap in either one with no other changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from memoryagent.retrieval.index import FaissIndex


@runtime_checkable
class RetrievalSampler(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        index: FaissIndex,
        k: int,
    ) -> tuple[torch.Tensor, np.ndarray, list[list[str]], torch.Tensor | None]:
        """Return (scores_unused, doc_idx, doc_texts, sample_logp).

        scores_unused: FAISS-side scores (detached, monitoring only)
        doc_idx: numpy [B, k] passage indices
        doc_texts: list[B][k] passage strings (for re-encoding by BGE)
        sample_logp: [B] or None — log-prob of the sampled action under the
                     sampling distribution (None for deterministic top-k)
        """
        ...


class TopKRetrieval:
    """Deterministic top-k. Used by replug_step today."""

    def __call__(
        self, q: torch.Tensor, index: FaissIndex, k: int,
    ) -> tuple[torch.Tensor, np.ndarray, list[list[str]], None]:
        scores, doc_idx, doc_texts = index.search(q, k)
        return torch.from_numpy(scores), doc_idx, doc_texts, None


class SampledRetrieval:
    """Sample k docs without replacement from softmax(scores/tau).

    NOT YET WIRED into a training step — designed for the upcoming PPO PR.
    """

    def __init__(self, tau: float = 1.0, candidate_pool: int = 100):
        self.tau = tau
        self.candidate_pool = candidate_pool

    def __call__(
        self, q: torch.Tensor, index: FaissIndex, k: int,
    ) -> tuple[torch.Tensor, np.ndarray, list[list[str]], torch.Tensor]:
        raise NotImplementedError(
            "SampledRetrieval is a stub for the PPO PR. Implement when wiring ppo_step."
        )
