"""Baseline: Qwen alone, no retrieval.

The floor we need to beat. Loads the chosen Qwen model (default 4B), runs
greedy generation on the NQ-open eval set with a system + user(question)
chat-templated prompt, and reports EM/F1 against the multi-answer gold.

Numbers from this script tell us how much of NQ-open the parametric weights
already know — anything REPLUG buys us has to come on top.

Usage:
    uv run python -m scripts.eval_baseline_qwen \
        --model Qwen/Qwen3.5-4B \
        --eval-size 200 \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from memoryagent.data.nq import load_nq
from memoryagent.device import resolve_device
from memoryagent.eval.qa_metrics import aggregate
from memoryagent.models.generator import QwenGenerator
from memoryagent.prompts import render_query_prompt


@torch.no_grad()
def generate_answer(
    gen: QwenGenerator,
    question: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = render_query_prompt(gen.tokenizer, question)
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
    eval_size: int,
    max_new_tokens: int,
    device_name: str,
    output_path: Path | None,
) -> None:
    ctx = resolve_device(device_name)
    print(f"device: {ctx.device}, dtype: {ctx.dtype_generator}")
    print(f"loading: {model_name}")
    gen = QwenGenerator(
        model_name,
        dtype=ctx.dtype_generator,
        attn_implementation=ctx.attn_implementation,
    ).to(ctx.device)
    gen.model.eval()

    print(f"loading NQ eval (limit={eval_size})...")
    _, _, eval_qa = load_nq(corpus_limit=1, train_limit=1, eval_limit=eval_size)
    print(f"  {len(eval_qa)} questions")

    preds: list[str] = []
    golds: list[list[str]] = []
    rows: list[dict] = []
    t0 = time.time()
    for ex in tqdm(eval_qa, desc="generate"):
        pred = generate_answer(
            gen, ex["question"], device=ctx.device, max_new_tokens=max_new_tokens,
        )
        preds.append(pred)
        gold_list = ex.get("answers") or [ex["answer"]]
        golds.append(gold_list)
        rows.append({"question": ex["question"], "gold": gold_list, "pred": pred})

    metrics = aggregate(preds, golds)
    elapsed = time.time() - t0
    print(
        f"\n[baseline qwen] model={model_name} n={metrics['n']} "
        f"EM={metrics['em']:.3f} F1={metrics['f1']:.3f} "
        f"({elapsed:.1f}s, {metrics['n'] / elapsed:.2f} q/s)"
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"model": model_name, "metrics": metrics, "rows": rows},
                f, indent=2,
            )
        print(f"saved: {output_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--output", type=Path, default=None, help="optional JSON dump of per-q rows")
    args = p.parse_args()
    main(
        model_name=args.model,
        eval_size=args.eval_size,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    cli()
