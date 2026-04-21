"""Baseline: Qwen + untrained BGE retrieval ("vanilla RAG").

Encodes the question with BGE directly (using its native query-instruction
prefix; NO projection from Qwen's hidden state), retrieves top-k passages
from a FAISS IndexFlatIP over the BGE-encoded corpus, stuffs them into the
user message, and runs greedy generation.

This is the second baseline. The gap from this to the trained REPLUG model
tells us how much joint training of the encoder + projection + generator
buys us over picking BGE off the shelf.

Usage:
    uv run python -m scripts.eval_baseline_rag \
        --model Qwen/Qwen3.5-4B \
        --corpus-size 10000 \
        --eval-size 200 \
        --k 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from memoryagent.data.nq import load_nq
from memoryagent.device import resolve_device
from memoryagent.eval.qa_metrics import aggregate
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.prompts import SYSTEM_PROMPT
from memoryagent.retrieval.index import FaissIndex


def render_rag_prompt(tokenizer, docs: list[str], question: str) -> str:
    """Chat-templated prompt with k stuffed docs.

    Different shape from the per-doc REPLUG prefix (one user message contains
    ALL k retrieved passages here, vs one per forward in training). Lives in
    the eval script because RAG-stuffing isn't part of the training-time
    prompt vocabulary in ``memoryagent.prompts``.
    """
    docs_block = "\n\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    user_content = f"Context:\n{docs_block}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _build_index(
    encoder: BGEEncoder,
    passages: list[dict],
    *,
    batch_size: int = 64,
) -> FaissIndex:
    texts = [p["text"] for p in passages]
    ids = [p["id"] for p in passages]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="encode corpus"):
            batch = texts[i : i + batch_size]
            embs = encoder.encode(batch, is_query=False).cpu().float().numpy()
            chunks.append(embs)
    embs = np.concatenate(chunks, axis=0)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    idx = FaissIndex(encoder.dim)
    idx.build(embs.astype(np.float32), ids, texts)
    return idx


@torch.no_grad()
def retrieve(
    encoder: BGEEncoder, question: str, index: FaissIndex, k: int,
) -> list[str]:
    q = encoder.encode([question], is_query=True)
    _, _, doc_texts = index.search(q, k)
    return doc_texts[0]


@torch.no_grad()
def generate_answer(
    gen: QwenGenerator,
    docs: list[str],
    question: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = render_rag_prompt(gen.tokenizer, docs, question)
    enc = gen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    out = gen.model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=gen.tokenizer.pad_token_id,
        eos_token_id=gen.tokenizer.eos_token_id,
    )
    new_tokens = out[0, enc["input_ids"].size(1):]
    return gen.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(
    *,
    model_name: str,
    corpus_size: int,
    eval_size: int,
    k: int,
    max_new_tokens: int,
    device_name: str,
    output_path: Path | None,
) -> None:
    ctx = resolve_device(device_name)
    print(f"device: {ctx.device}, dtype: {ctx.dtype_generator}")

    print(f"loading: {model_name}, BAAI/bge-small-en-v1.5")
    gen = QwenGenerator(
        model_name,
        dtype=ctx.dtype_generator,
        attn_implementation=ctx.attn_implementation,
    ).to(ctx.device)
    gen.model.eval()
    encoder = BGEEncoder().to(ctx.device)
    encoder.eval()

    print(f"loading NQ (corpus={corpus_size}, eval={eval_size})...")
    passages, _, eval_qa = load_nq(
        corpus_limit=corpus_size, train_limit=1, eval_limit=eval_size,
    )
    print(f"  passages={len(passages)} eval={len(eval_qa)}")

    index = _build_index(encoder, passages)
    print(f"  index built: size={len(index)}")

    preds: list[str] = []
    golds: list[list[str]] = []
    rows: list[dict] = []
    t0 = time.time()
    for ex in tqdm(eval_qa, desc="rag-generate"):
        docs = retrieve(encoder, ex["question"], index, k)
        pred = generate_answer(
            gen, docs, ex["question"],
            device=ctx.device, max_new_tokens=max_new_tokens,
        )
        preds.append(pred)
        gold_list = ex.get("answers") or [ex["answer"]]
        golds.append(gold_list)
        rows.append({
            "question": ex["question"], "gold": gold_list, "pred": pred,
            "docs": docs,
        })

    metrics = aggregate(preds, golds)
    elapsed = time.time() - t0
    print(
        f"\n[baseline rag] model={model_name} k={k} corpus={len(passages)} "
        f"n={metrics['n']} EM={metrics['em']:.3f} F1={metrics['f1']:.3f} "
        f"({elapsed:.1f}s, {metrics['n'] / elapsed:.2f} q/s)"
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "k": k,
                    "corpus_size": len(passages),
                    "metrics": metrics,
                    "rows": rows,
                },
                f, indent=2,
            )
        print(f"saved: {output_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--output", type=Path, default=None, help="optional JSON dump of per-q rows")
    args = p.parse_args()
    main(
        model_name=args.model,
        corpus_size=args.corpus_size,
        eval_size=args.eval_size,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    cli()
