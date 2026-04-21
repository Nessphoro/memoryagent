# memoryagent

Joint REPLUG-style training of a dense retriever (`BAAI/bge-small-en-v1.5`) and
a small generator (`Qwen/Qwen3.5-*B`). The generator's last-token hidden state
(post-final-RMSNorm, pre-LM-head) is projected into BGE space; top-k passages
are retrieved from a FAISS `IndexFlatIP`; gold-answer NLL is back-propagated
through the projection into both the encoder and the generator. The
gradient-bearing dot product is recomputed in torch over the retrieved k —
FAISS is used purely as a non-differentiable kNN data structure.

## Algorithm (one step)

```
question
  → generator forward, last hidden state           h    [B, 1024]
  → projection (Linear → LayerNorm → L2)           q    [B, 384]
  → FAISS IndexFlatIP search(q, k)                 doc_idx, doc_texts
  → BGE re-encode top-k (gradient-bearing)         E    [B, k, 384]
  → scores = einsum("bd,bkd->bk", q, E)            scores [B, k]
  → log_w = log_softmax(scores / tau)              log_w  [B, k]
  → for each doc: gen forward on chat-templated
        [system + user(doc + question) + answer]
  → gather log p_i(y_t | y_<t, doc_i) over gold y
  → log P_mix(y_t) = logsumexp_i (log_w_i + log p_i)
  → loss = -mean_t (log P_mix * answer_mask)
```

Gradients reach the generator (per-doc forwards), the projection (via the
torch-recomputed scores), and the BGE encoder (via the query-side forward and
the re-encoded top-k). Index-side embeddings are a frozen snapshot, refreshed
every N steps with the up-to-date encoder.

## Layout

```
src/memoryagent/
  config.py              pydantic dataclasses, YAML loader
  prompts.py             Qwen chat-template renderers (query, doc-qa prefix)
  device.py              dtype/ckpt/8-bit/flash flags by device
  models/                generator (Qwen), encoder (BGE), projection
  retrieval/             FaissIndex, IndexRefresher, encode_corpus
  data/                  toy fixture, BeIR/nq loader, collate
  training/              replug step, losses, optim, muon helpers, loop
  eval/                  retrieval metrics (recall@k, MRR), QA metrics (EM, F1)
  ppo/                   stubs for the planned PPO stage
scripts/                 train.py, smoke.py, encode_corpus.py, eval_baseline_*.py
configs/                 smoke.yaml, m3_proto.yaml, m3_nq.yaml, rtx4090.yaml
tests/                   pytest
```

## Quick start

```bash
uv sync                                        # install deps (torch >= 2.11 for Muon)
uv run python -m scripts.smoke                 # 100-doc end-to-end smoke
uv run pytest                                  # unit tests

# Real training run (NQ-open + BeIR/nq corpus)
uv run python -m scripts.train --config configs/m3_nq.yaml
uv run python -m scripts.train --config configs/rtx4090.yaml --resume runs/.../ckpt_*.pt
```

## Configs

| Config              | Hardware             | Model              | Notes                                  |
|---------------------|----------------------|--------------------|----------------------------------------|
| `smoke.yaml`        | any                  | Qwen3.5-0.8B       | 100-doc toy, 50 steps, sanity check    |
| `m3_proto.yaml`     | M3 Max               | Qwen3.5-0.8B       | Toy data, longer run                   |
| `m3_nq.yaml`        | M3 Max (MPS)         | Qwen3.5-0.8B       | NQ-open + 2k BeIR passages, AdamW      |
| `rtx4090.yaml`      | RTX 4090 (CUDA)      | Qwen3.5-1.7B       | NQ + 100k passages, Muon + Adam8bit, ~15 GB |

## Optimizer

Two paths via `optim.optimizer`:

- `adamw` (default) — single `torch.optim.AdamW` (or `AdamW8bit` on CUDA when
  `use_8bit_adam=true`) with three per-module param groups.
- `muon` — `torch.optim.Muon` over inner 2D weights chained with AdamW over
  1D params (norms, biases) and embedding-like 2D matrices (`embed_tokens`,
  `lm_head`, BERT `embeddings.*`). Defaults to `adjust_lr_fn="match_rms_adamw"`
  so the same per-module LR carries over without a hand-tuned multiplier.

Requires PyTorch >= 2.11 for the built-in `torch.optim.Muon`.

## Eval

Two reference baselines to anchor what training has to beat:

```bash
# Qwen alone, no retrieval — parametric-knowledge floor
uv run python -m scripts.eval_baseline_qwen \
    --model Qwen/Qwen3.5-4B --eval-size 200

# Qwen + BGE off-the-shelf retrieval (no projection) — vanilla RAG
uv run python -m scripts.eval_baseline_rag \
    --model Qwen/Qwen3.5-4B --corpus-size 10000 --eval-size 200 --k 3
```

Both report exact-match and token-F1 against multi-answer NQ-open gold via
`memoryagent.eval.qa_metrics` (SQuAD-style normalize, max over gold list).

## Notes

- FAISS is the kNN data structure; the gradient-bearing dot product is
  recomputed in torch over the retrieved k. Never use FAISS-returned scores
  in the loss — they're detached numpy.
- The query-side forward uses `output_hidden_states=True` and is kept
  separate from the per-doc answer-scoring pass (doubling activations on the
  query forward would blow memory on MPS).
- Both query-encoding and per-doc forwards go through Qwen's chat template
  (`system + user + assistant opener`); the answer is appended without a
  leading space because the assistant opener ends with `\n`.
- macOS quirks: `attn_implementation="sdpa"` (never `flash_attention_2`),
  `faiss.omp_set_num_threads(1)` to avoid the libomp double-load segfault,
  no `pin_memory`, `num_workers=0`.
