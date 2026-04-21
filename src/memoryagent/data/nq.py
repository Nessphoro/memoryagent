"""NQ-open questions + BeIR/nq passages, with on-disk caching.

Pivoted away from facebook/wiki_dpr and Tevatron/wikipedia-nq-corpus — both
use legacy Python loading scripts which are unsupported in datasets>=3.0.
BeIR/nq is parquet-native and loads cleanly. ~2.7M Wikipedia passages.

We use BeIR's corpus for the candidate pool and nq_open's question+answer
pairs for the REPLUG training signal. We do NOT pull BeIR's qrels — the
question text in nq_open and BeIR queries don't always align by ID, and we
don't need gold passage labels for the REPLUG loss (only for retrieval-eval,
which we approximate via answer-string-in-retrieved-passages as a proxy).

Streamed + take first N → cache to JSONL under `.cache/nq/`. Re-running with
the same N is a JSONL read — no HF round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".cache/nq")


def load_nq(
    *,
    corpus_limit: int = 10_000,
    train_limit: int | None = 1_000,
    eval_limit: int | None = 100,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (passages, train_qa, eval_qa) in the load_dataset standard format."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    passages = _load_passages(corpus_limit, cache_dir)
    train_qa = _load_nq_open("train", train_limit, cache_dir)
    eval_qa = _load_nq_open("validation", eval_limit, cache_dir)
    return passages, train_qa, eval_qa


def _load_passages(limit: int, cache_dir: Path) -> list[dict]:
    cache = cache_dir / f"beir_nq_passages_{limit}.jsonl"
    if cache.exists():
        return _read_jsonl(cache)
    from datasets import load_dataset

    print(f"streaming BeIR/nq (corpus), taking first {limit:,}...")
    ds = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
    passages: list[dict] = []
    for ex in ds:
        if len(passages) >= limit:
            break
        # BeIR schema: _id, title, text
        doc_id = ex.get("_id") or ex.get("id") or ex.get("docid")
        title = (ex.get("title") or "").strip()
        text = (ex.get("text") or "").strip()
        body = f"{title}. {text}" if title else text
        passages.append({"id": str(doc_id), "text": body})
    _write_jsonl(cache, passages)
    print(f"  cached {len(passages):,} passages → {cache}")
    return passages


def _load_nq_open(split: str, limit: int | None, cache_dir: Path) -> list[dict]:
    suffix = "all" if limit is None else str(limit)
    cache = cache_dir / f"nq_open_{split}_{suffix}.jsonl"
    if cache.exists():
        return _read_jsonl(cache)
    from datasets import load_dataset

    print(f"loading google-research-datasets/nq_open ({split}, limit={limit})...")
    ds = load_dataset("google-research-datasets/nq_open", split=split)
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))
    qa: list[dict] = []
    for ex in ds:
        answers = list(ex.get("answer") or [])
        qa.append({
            "question": ex["question"],
            "answer": answers[0] if answers else "",
            "answers": answers,
            "gold_doc_id": None,
        })
    _write_jsonl(cache, qa)
    print(f"  cached {len(qa):,} {split} QA pairs → {cache}")
    return qa


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
