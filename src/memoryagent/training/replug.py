from __future__ import annotations

import torch
import torch.nn.functional as F

from memoryagent.data.collate import QABatch
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.prompts import render_doc_qa_prefix
from memoryagent.retrieval.index import FaissIndex
from memoryagent.training.losses import mixture_nll


def _truncate_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate `text` so it tokenizes to at most `max_tokens` Qwen tokens."""
    if max_tokens <= 0:
        return ""
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=max_tokens,
    ).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_replug_inputs(
    doc_texts: list[list[str]],
    question_texts: list[str],
    answer_texts: list[str],
    tokenizer,
    max_length: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat ``[B*k, T_max]`` padded inputs of chat-templated ``[prefix || answer]``.

    Tokenization strategy:
      - Tokenize the answer ONCE per batch element (no leading space — the asst
        opener ends with ``\\n``) so the gathered log-probs correspond to the
        SAME tokens y_t across all k docs. Without this, the mixture would be
        ill-defined.
      - Render each prefix through Qwen's chat template via
        ``render_doc_qa_prefix(tokenizer, doc, question)``.
      - The prefix-prefix (system + user envelope) is fixed per question, so
        the only variable-length part is the doc text. We truncate the doc
        text to fit the budget — never the templated string itself, since
        right-truncation would chop the asst opener and left-truncation would
        chop the system prompt, both of which break the format.

    Returns: ``(input_ids [B*k, T_max], attention_mask [B*k, T_max],
              prefix_lens [B*k] int64, ans_lens [B*k] int64)``.
    """
    B = len(question_texts)
    k = len(doc_texts[0])
    pad_id = tokenizer.pad_token_id

    flat: list[list[int]] = []
    prefix_lens_list: list[int] = []
    ans_lens_list: list[int] = []

    for b in range(B):
        q = question_texts[b]
        a = answer_texts[b]
        ans_ids = tokenizer(a, add_special_tokens=False).input_ids
        # Bound the doc text so the templated prefix + answer fits in max_length.
        # The empty-doc render gives us the exact non-doc overhead (system
        # prompt + user envelope + question + asst opener); 8-token slack
        # covers BPE boundary effects when the doc is spliced in.
        empty_prefix_ids = tokenizer(
            render_doc_qa_prefix(tokenizer, "", q), add_special_tokens=False,
        ).input_ids
        doc_budget = max(1, max_length - len(empty_prefix_ids) - len(ans_ids) - 8)
        for i in range(k):
            d = _truncate_text_to_tokens(tokenizer, doc_texts[b][i], doc_budget)
            prefix_text = render_doc_qa_prefix(tokenizer, d, q)
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
            full = prefix_ids + ans_ids
            assert len(full) <= max_length, (
                f"build_replug_inputs: budget exhausted "
                f"(len={len(full)} > max_length={max_length}); "
                f"bump max_length or shrink max_passage_tokens / max_question_tokens."
            )
            flat.append(full)
            prefix_lens_list.append(len(prefix_ids))
            ans_lens_list.append(len(ans_ids))

    T_max = max(len(f) for f in flat)
    input_ids = torch.full((B * k, T_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B * k, T_max), dtype=torch.long)
    for j, ids in enumerate(flat):
        L = len(ids)
        input_ids[j, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[j, :L] = 1

    prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.long)
    ans_lens = torch.tensor(ans_lens_list, dtype=torch.long)
    return input_ids, attention_mask, prefix_lens, ans_lens


def gather_answer_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    ans_lens: torch.Tensor,
    B: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather log p(y_t | y_<t) for the gold answer tokens, memory-efficient.

    Naive: log_softmax over the full ``[B*k, T, V]`` logits and then index. With
    Qwen3.5's 248K vocab and B*k=10 sequences of T~300, that's ~3GB of fp32
    held in the autograd graph until backward — which is what was OOM-ing on
    MPS. Instead we slice logits to only the prediction positions
    ``[B*k, T_ans, V]`` (~50MB), do log_softmax there, then gather target tokens.

    Position ``prefix_lens[j]`` is the FIRST answer token y_0; the model's
    prediction for y_t lives at position ``prefix_lens[j] + t - 1`` in logits.

    Returns:
        gathered:    [B, k, T_ans] — log-probs of each gold token under each doc.
        answer_mask: [B, T_ans]    — True where the position is a valid answer slot
                                     (constant across k since ans_len is constant per b).
    """
    BK, T_max, _ = logits.shape
    T_ans = int(ans_lens.max().item())
    device = logits.device

    t_arange = torch.arange(T_ans, device=device)
    answer_pos = prefix_lens.unsqueeze(-1) + t_arange         # [BK, T_ans] target positions
    answer_pos_clamped = answer_pos.clamp(max=T_max - 1)
    pred_pos = (answer_pos_clamped - 1).clamp(min=0)

    targets = torch.gather(input_ids, 1, answer_pos_clamped)  # [BK, T_ans]

    # Slice logits to prediction positions only: [BK, T_ans, V] instead of [BK, T_max, V].
    bk_idx = torch.arange(BK, device=device).unsqueeze(-1).expand(-1, T_ans)
    logits_at_pred = logits[bk_idx, pred_pos]                  # [BK, T_ans, V]

    # log_softmax on the small slice, in fp32 for numerical stability.
    log_probs = torch.log_softmax(logits_at_pred.float(), dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [BK, T_ans]

    answer_mask_flat = t_arange.unsqueeze(0) < ans_lens.unsqueeze(-1)  # [BK, T_ans]
    gathered = gathered.view(B, k, T_ans)
    answer_mask = answer_mask_flat.view(B, k, T_ans)[:, 0, :]
    return gathered, answer_mask


def replug_step(
    batch: QABatch,
    generator: QwenGenerator,
    encoder: BGEEncoder,
    projection: Projection,
    index: FaissIndex,
    *,
    k: int,
    tau: float,
    max_length: int = 1024,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One REPLUG-LSR training step. See plan §3 for the full math."""
    B = batch.size
    device = next(generator.model.parameters()).device

    # 1. Query representation from the GENERATOR — gradient flows to gen + proj.
    q_hidden = generator.encode_query(batch.question_ids, batch.question_mask)  # [B, H]
    q_hidden = q_hidden.float()  # projection is fp32
    q = projection(q_hidden)                                                     # [B, dim], L2-normed

    # 2. FAISS kNN over the stale snapshot — non-differentiable, returns texts.
    _, _, doc_texts = index.search(q, k)

    # 3. Re-encode top-k passages with the CURRENT BGE encoder for fresh embeddings.
    flat_texts = [t for row in doc_texts for t in row]
    fresh_doc_embs = encoder.encode(flat_texts, is_query=False)                  # [B*k, dim]
    fresh_doc_embs = fresh_doc_embs.view(B, k, -1)                               # [B, k, dim]

    # 4. Gradient-bearing scores: flows to BOTH q (gen+proj) AND doc_embs (encoder).
    scores = torch.einsum("bd,bkd->bk", q, fresh_doc_embs)                       # [B, k]
    log_w = torch.log_softmax(scores / tau, dim=-1)                              # [B, k]

    # 5. Build flat [B*k, T] generator inputs of [doc || question || answer].
    input_ids, attention_mask, prefix_lens, ans_lens = build_replug_inputs(
        doc_texts,
        batch.question_texts,
        batch.answer_texts,
        generator.tokenizer,
        max_length=max_length,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prefix_lens = prefix_lens.to(device)
    ans_lens = ans_lens.to(device)

    # 6. Generator forward over B*k sequences. We do NOT materialize a full
    #    [B*k, T, V=248K] log_softmax tensor — see gather_answer_logprobs.
    out = generator.lm_forward(input_ids, attention_mask=attention_mask)

    # 7. Gather log p_i(y_t) for the gold answer tokens (memory-efficient slice).
    gathered, answer_mask = gather_answer_logprobs(
        out.logits, input_ids, prefix_lens, ans_lens, B, k,
    )                                                                            # [B, k, T_ans], [B, T_ans]

    # 8. Mixture NLL via logsumexp.
    loss = mixture_nll(gathered, log_w, answer_mask)

    metrics = {
        "loss": loss.detach(),
        "scores_top1": scores.max(dim=-1).values.mean().detach(),
        "weight_entropy": (-(log_w.exp() * log_w).sum(-1)).mean().detach(),
    }
    return loss, metrics
