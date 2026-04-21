from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from memoryagent.retrieval.index import FaissIndex


def _normed(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_topk_matches_brute_force():
    rng = np.random.default_rng(0)
    n, d, k = 50, 16, 5
    embs = _normed(rng.standard_normal((n, d)).astype(np.float32))
    ids = [f"doc-{i}" for i in range(n)]
    texts = [f"text {i}" for i in range(n)]

    idx = FaissIndex(d)
    idx.build(embs, ids, texts)
    assert len(idx) == n
    assert idx.version == 1

    queries = _normed(rng.standard_normal((4, d)).astype(np.float32))
    q_torch = torch.from_numpy(queries)
    faiss_scores, doc_idx, doc_texts = idx.search(q_torch, k)

    # Brute force ground truth
    brute_scores = queries @ embs.T
    brute_topk_idx = np.argsort(-brute_scores, axis=1)[:, :k]

    assert doc_idx.shape == (4, k)
    np.testing.assert_array_equal(doc_idx, brute_topk_idx)

    expected_scores = np.take_along_axis(brute_scores, brute_topk_idx, axis=1)
    np.testing.assert_allclose(faiss_scores, expected_scores, rtol=1e-5, atol=1e-5)

    # doc_texts must align with doc_idx
    for b in range(4):
        for j in range(k):
            assert doc_texts[b][j] == texts[doc_idx[b, j]]


def test_rebuild_increments_version_and_swaps_data():
    rng = np.random.default_rng(1)
    d = 8
    embs1 = _normed(rng.standard_normal((10, d)).astype(np.float32))
    embs2 = _normed(rng.standard_normal((20, d)).astype(np.float32))

    idx = FaissIndex(d)
    idx.build(embs1, [f"a{i}" for i in range(10)], [f"a{i}" for i in range(10)])
    assert len(idx) == 10 and idx.version == 1

    idx.build(embs2, [f"b{i}" for i in range(20)], [f"b{i}" for i in range(20)])
    assert len(idx) == 20 and idx.version == 2
    assert idx.ids[0] == "b0"


def test_save_load_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(2)
    d, n, k = 12, 30, 3
    embs = _normed(rng.standard_normal((n, d)).astype(np.float32))
    ids = [f"x{i}" for i in range(n)]
    texts = [f"passage number {i}" for i in range(n)]

    idx = FaissIndex(d)
    idx.build(embs, ids, texts)

    emb_path = tmp_path / "embeddings_v1.faiss"
    ids_path = tmp_path / "ids_v1.json"
    idx.save(emb_path, ids_path)

    loaded = FaissIndex.load(emb_path, ids_path)
    assert len(loaded) == n
    assert loaded.ids == ids
    assert loaded.texts == texts
    assert loaded.version == 1

    q = torch.from_numpy(_normed(rng.standard_normal((2, d)).astype(np.float32)))
    s1, i1, _ = idx.search(q, k)
    s2, i2, _ = loaded.search(q, k)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, rtol=1e-6, atol=1e-6)
