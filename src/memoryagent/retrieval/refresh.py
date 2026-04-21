from __future__ import annotations

import numpy as np
import torch

from memoryagent.models.encoder import BGEEncoder
from memoryagent.retrieval.index import FaissIndex


def encode_corpus(
    encoder: BGEEncoder,
    texts: list[str],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Re-encode every passage in `texts` with the current encoder, no_grad, eval mode."""
    was_train = encoder.training
    encoder.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            embs = encoder.encode(chunk, is_query=False).cpu().float().numpy()
            out.append(embs)
    if was_train:
        encoder.train()
    arr = np.concatenate(out, axis=0)
    arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr.astype(np.float32)


class IndexRefresher:
    """Re-encodes the corpus and atomically swaps the FAISS index every N steps.

    The encoder learns to produce queries that match the *snapshot*. Refreshing
    too often makes the snapshot move under the optimizer's feet → training
    oscillates. Refresh cadence is the load-bearing knob.
    """

    def __init__(
        self,
        encoder: BGEEncoder,
        passages: list[dict],
        *,
        refresh_every: int,
        encode_batch_size: int = 64,
    ):
        self.encoder = encoder
        self.ids = [p["id"] for p in passages]
        self.texts = [p["text"] for p in passages]
        self.refresh_every = refresh_every
        self.encode_batch_size = encode_batch_size
        # First call to maybe_refresh always fires.
        self.last_refresh_step: int | None = None

    def maybe_refresh(self, index: FaissIndex, step: int) -> bool:
        if (
            self.last_refresh_step is not None
            and step - self.last_refresh_step < self.refresh_every
        ):
            return False
        self.refresh(index)
        self.last_refresh_step = step
        return True

    def refresh(self, index: FaissIndex) -> None:
        embs = encode_corpus(
            self.encoder, self.texts, batch_size=self.encode_batch_size,
        )
        index.build(embs, self.ids, self.texts)
