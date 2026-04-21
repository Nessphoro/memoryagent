from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch

# macOS: PyTorch and FAISS each link their own libomp. Even with
# KMP_DUPLICATE_LIB_OK=TRUE, FAISS calls can segfault when both runtimes
# coexist. Forcing FAISS to use a single thread keeps it off the OpenMP
# code path and is plenty fast for our workloads (1M passages, k=5).
faiss.omp_set_num_threads(1)


class FaissIndex:
    """Cosine-similarity (inner product on L2-normed vectors) FAISS index.

    Stores passage texts alongside for per-step re-encoding. The FAISS scores
    are detached numpy and used only for candidate selection — the gradient-
    bearing scores are recomputed in torch by feeding the top-k passage texts
    back through the BGE encoder. The index itself owns no torch tensors.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.version: int = 0

    def __len__(self) -> int:
        return self.faiss_index.ntotal

    def build(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        texts: list[str],
    ) -> None:
        """Initial build or refresh. Embeddings must be L2-normalized [N, dim]."""
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings dim {embeddings.shape[1]} != index dim {self.dim}")
        if len(ids) != len(texts) or len(ids) != embeddings.shape[0]:
            raise ValueError("ids, texts, and embeddings must have the same length")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings)
        new_index = faiss.IndexFlatIP(self.dim)
        new_index.add(embeddings)
        self.faiss_index = new_index
        self.ids = list(ids)
        self.texts = list(texts)
        self.version += 1

    def search(
        self,
        q: torch.Tensor,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """kNN over the stale snapshot. Returns (faiss_scores, doc_idx, doc_texts).

        - faiss_scores: [B, k] fp32, NOT differentiable; for monitoring only.
        - doc_idx:      [B, k] int64.
        - doc_texts:    list[B][k] strings — feed to the BGE encoder for fresh
                        embeddings to compute the gradient-bearing scores.
        """
        if k > len(self):
            raise ValueError(f"k={k} > index size {len(self)}")
        q_np = q.detach().to("cpu", dtype=torch.float32).numpy()
        if not q_np.flags["C_CONTIGUOUS"]:
            q_np = np.ascontiguousarray(q_np)
        scores, doc_idx = self.faiss_index.search(q_np, k)
        doc_texts = [[self.texts[i] for i in row] for row in doc_idx]
        return scores, doc_idx, doc_texts

    def reconstruct(self, idx: int) -> np.ndarray:
        """Return the stored embedding for index `idx`. Used by tests."""
        return self.faiss_index.reconstruct(int(idx))

    def save(self, embeddings_path: Path, ids_path: Path) -> None:
        embeddings_path = Path(embeddings_path)
        ids_path = Path(ids_path)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(embeddings_path))
        with open(ids_path, "w") as f:
            json.dump(
                {"ids": self.ids, "texts": self.texts, "version": self.version},
                f,
            )

    @classmethod
    def load(cls, embeddings_path: Path, ids_path: Path) -> FaissIndex:
        idx = faiss.read_index(str(embeddings_path))
        with open(ids_path) as f:
            data = json.load(f)
        obj = cls(idx.d)
        obj.faiss_index = idx
        obj.ids = list(data["ids"])
        obj.texts = list(data["texts"])
        obj.version = int(data.get("version", 0))
        return obj
