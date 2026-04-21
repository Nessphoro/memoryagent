from __future__ import annotations

from collections.abc import Iterator

import torch

from memoryagent.data.collate import QABatch
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.retrieval.index import FaissIndex


@torch.no_grad()
def retrieval_metrics(
    eval_iter: Iterator[QABatch],
    generator: QwenGenerator,
    projection: Projection,
    index: FaissIndex,
    *,
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Recall@k and MRR over an eval iterator.

    For each batch element, uses the gold_doc_id if available, otherwise falls
    back to "is the gold answer string contained in any retrieved passage?" as
    a noisy heuristic for retrieval quality (useful when gold passage labels
    aren't part of the dataset, e.g. nq_open).
    """
    was_train_gen = generator.training
    was_train_proj = projection.training
    generator.eval()
    projection.eval()

    n = 0
    recall_hits = 0
    mrr_sum = 0.0
    mode_counts = {"id": 0, "string": 0}
    for batch in eval_iter:
        batch = batch.to(device)
        h = generator.encode_query(batch.question_ids, batch.question_mask)
        q = projection(h.float())
        _, doc_idx, _ = index.search(q, k)
        gold_ids = batch.gold_doc_ids or [None] * batch.size
        for b in range(batch.size):
            n += 1
            retrieved_ids = [index.ids[int(doc_idx[b, j])] for j in range(k)]
            gold = gold_ids[b]
            if gold is not None:
                mode_counts["id"] += 1
                if gold in retrieved_ids:
                    recall_hits += 1
                    mrr_sum += 1.0 / (retrieved_ids.index(gold) + 1)
            else:
                mode_counts["string"] += 1
                ans = batch.answer_texts[b].strip().lower()
                if not ans:
                    continue
                retrieved_texts = [index.texts[int(doc_idx[b, j])].lower() for j in range(k)]
                hit_rank = next(
                    (j + 1 for j, t in enumerate(retrieved_texts) if ans in t),
                    None,
                )
                if hit_rank is not None:
                    recall_hits += 1
                    mrr_sum += 1.0 / hit_rank

    if was_train_gen:
        generator.train()
    if was_train_proj:
        projection.train()

    if n == 0:
        return {f"recall@{k}": 0.0, "mrr": 0.0, "n": 0}
    return {
        f"recall@{k}": recall_hits / n,
        "mrr": mrr_sum / n,
        "n": n,
        "eval_mode": "id" if mode_counts["id"] > mode_counts["string"] else "string",
    }
