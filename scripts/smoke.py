"""End-to-end smoke test: 100-doc toy corpus, 10 QA, k=2, 50 steps.

Asserts that training reduces loss meaningfully and that retrieval recall
does not regress. Runs end-to-end in <5 min on M3 Max bf16, faster on CUDA.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from memoryagent.data.collate import QABatch
from memoryagent.data.toy import PASSAGES, QA_PAIRS
from memoryagent.device import resolve_device
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.retrieval.index import FaissIndex
from memoryagent.seed import seed_all
from memoryagent.training.replug import replug_step


@dataclass
class SmokeResult:
    initial_loss: float
    final_loss: float
    initial_recall_at_k: float
    final_recall_at_k: float
    elapsed_seconds: float


def _build_initial_index(encoder: BGEEncoder) -> FaissIndex:
    texts = [p.text for p in PASSAGES]
    ids = [p.id for p in PASSAGES]
    with torch.no_grad():
        embs = encoder.encode(texts, is_query=False).cpu().float().numpy()
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    idx = FaissIndex(encoder.dim)
    idx.build(embs, ids, texts)
    return idx


def _make_batch(qa_subset, gen_tokenizer) -> QABatch:
    questions = [qa.question for qa in qa_subset]
    answers = [qa.answer for qa in qa_subset]
    enc = gen_tokenizer(questions, padding=True, return_tensors="pt")
    return QABatch(
        question_ids=enc["input_ids"],
        question_mask=enc["attention_mask"],
        question_texts=questions,
        answer_texts=answers,
        gold_doc_ids=[qa.gold_doc_id for qa in qa_subset],
    )


def _eval_recall_at_k(
    generator: QwenGenerator,
    encoder: BGEEncoder,
    projection: Projection,
    index: FaissIndex,
    k: int,
) -> float:
    questions = [qa.question for qa in QA_PAIRS]
    gold_ids = [qa.gold_doc_id for qa in QA_PAIRS]
    enc = generator.tokenizer(questions, padding=True, return_tensors="pt")
    device = next(generator.model.parameters()).device
    q_ids = enc["input_ids"].to(device)
    q_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        h = generator.encode_query(q_ids, q_mask)
        q = projection(h.float())
    _, doc_idx, _ = index.search(q, k)
    hits = sum(
        1
        for i, g in enumerate(gold_ids)
        if g in [index.ids[int(doc_idx[i, j])] for j in range(k)]
    )
    return hits / len(gold_ids)


def run_smoke(
    *,
    seed: int = 42,
    batch_size: int = 2,
    k: int = 2,
    tau: float = 0.1,
    max_steps: int = 50,
    lr_generator: float = 5e-6,
    lr_encoder: float = 5e-5,
    lr_projection: float = 1e-3,
    max_length: int = 256,
) -> SmokeResult:
    seed_all(seed)
    ctx = resolve_device("auto")
    print(f"device: {ctx.device}, dtype_gen={ctx.dtype_generator}")

    print("loading models...")
    generator = QwenGenerator(dtype=ctx.dtype_generator).to(ctx.device)
    encoder = BGEEncoder().to(ctx.device)
    projection = Projection(generator.hidden_size, encoder.dim).to(ctx.device)

    print("building initial index...")
    index = _build_initial_index(encoder)
    print(f"  index size: {len(index)}")

    optim = torch.optim.AdamW(
        [
            {"params": generator.parameters(), "lr": lr_generator},
            {"params": encoder.parameters(), "lr": lr_encoder},
            {"params": projection.parameters(), "lr": lr_projection},
        ],
        weight_decay=0.01,
    )

    initial_recall = _eval_recall_at_k(generator, encoder, projection, index, k=k)
    print(f"initial recall@{k}: {initial_recall:.2%}")

    losses: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(max_steps):
        qa_subset = rng.sample(QA_PAIRS, batch_size)
        batch = _make_batch(qa_subset, generator.tokenizer).to(ctx.device)

        loss, metrics = replug_step(
            batch,
            generator,
            encoder,
            projection,
            index,
            k=k,
            tau=tau,
            max_length=max_length,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                *generator.parameters(),
                *encoder.parameters(),
                *projection.parameters(),
            ],
            max_norm=1.0,
        )
        optim.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"step {step + 1:3d}/{max_steps}: "
                f"loss={loss.item():.4f}  top1={metrics['scores_top1'].item():.3f}  "
                f"H[w]={metrics['weight_entropy'].item():.3f}"
            )

    elapsed = time.time() - t0
    final_recall = _eval_recall_at_k(generator, encoder, projection, index, k=k)

    initial_loss = float(np.mean(losses[:5]))
    final_loss = float(np.mean(losses[-5:]))
    print(f"trained in {elapsed:.1f}s")
    print(f"avg loss first 5: {initial_loss:.4f}  |  avg loss last 5: {final_loss:.4f}  "
          f"({(1 - final_loss / max(initial_loss, 1e-9)):.1%} reduction)")
    print(f"recall@{k}: {initial_recall:.2%} → {final_recall:.2%}")

    return SmokeResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_recall_at_k=initial_recall,
        final_recall_at_k=final_recall,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    result = run_smoke()
    assert result.final_loss < 0.7 * result.initial_loss, (
        f"loss did not decrease enough: {result.final_loss / result.initial_loss:.2f}"
    )
    assert result.final_recall_at_k >= result.initial_recall_at_k, (
        f"recall regressed: {result.initial_recall_at_k:.2%} → {result.final_recall_at_k:.2%}"
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
