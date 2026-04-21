from __future__ import annotations

import random
from collections.abc import Iterator

from memoryagent.config import DataConfig
from memoryagent.data.collate import QABatch, collate_qa


def load_dataset(cfg: DataConfig) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (passages, train_qa, eval_qa) as plain dicts.

    - passages: list of {"id": str, "text": str}
    - train_qa, eval_qa: list of {"question": str, "answer": str, "gold_doc_id": str | None}

    `cfg.corpus_subsample`, `cfg.train_size`, `cfg.eval_size` are pushed down to
    the loader so we don't pull more data than needed (matters for wiki_dpr).
    """
    if cfg.name == "toy":
        from memoryagent.data.toy import PASSAGES, QA_PAIRS

        passages = [{"id": p.id, "text": p.text} for p in PASSAGES]
        qa = [
            {"question": q.question, "answer": q.answer, "gold_doc_id": q.gold_doc_id}
            for q in QA_PAIRS
        ]
        return passages, qa, qa
    if cfg.name == "nq":
        from memoryagent.data.nq import load_nq

        return load_nq(
            corpus_limit=cfg.corpus_subsample if cfg.corpus_subsample is not None else 10_000,
            train_limit=cfg.train_size,
            eval_limit=cfg.eval_size,
        )
    raise ValueError(f"Unknown dataset: {cfg.name!r}")


def infinite_train_iterator(
    qa_examples: list[dict],
    tokenizer,
    *,
    batch_size: int,
    seed: int,
    max_question_tokens: int = 64,
) -> Iterator[QABatch]:
    """Yield random batches forever (sampling with replacement when batch > set)."""
    rng = random.Random(seed)
    n = len(qa_examples)
    while True:
        if n >= batch_size:
            chunk = rng.sample(qa_examples, batch_size)
        else:
            chunk = [rng.choice(qa_examples) for _ in range(batch_size)]
        yield collate_qa(chunk, tokenizer, max_question_tokens=max_question_tokens)


def eval_iterator(
    qa_examples: list[dict],
    tokenizer,
    *,
    batch_size: int,
    max_question_tokens: int = 64,
) -> Iterator[QABatch]:
    """Single pass over the eval set in fixed order."""
    for i in range(0, len(qa_examples), batch_size):
        chunk = qa_examples[i : i + batch_size]
        yield collate_qa(chunk, tokenizer, max_question_tokens=max_question_tokens)
