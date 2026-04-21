from __future__ import annotations

import numpy as np
import pytest
import torch

from memoryagent.data.collate import collate_qa
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.retrieval.index import FaissIndex
from memoryagent.training.replug import (
    build_replug_inputs,
    gather_answer_logprobs,
    replug_step,
)


def _normed(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def models():
    """Load real models once for the module — slow but needed to test the pipeline."""
    gen = QwenGenerator(dtype=torch.float32)  # fp32 for cleaner gradient checks on CPU
    enc = BGEEncoder()
    proj = Projection(gen.hidden_size, enc.dim)
    return gen, enc, proj


def _build_toy_index(encoder: BGEEncoder, n: int = 8) -> FaissIndex:
    texts = [
        "The Eiffel Tower is located in Paris, France.",
        "William Shakespeare wrote the play Hamlet.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Python is a high-level programming language.",
        "Mount Everest is the tallest mountain in the world.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Albert Einstein developed the theory of relativity.",
        "The mitochondrion is the powerhouse of the cell.",
    ][:n]
    ids = [f"doc-{i}" for i in range(n)]
    with torch.no_grad():
        embs = encoder.encode(texts, is_query=False).cpu().float().numpy()
    embs = _normed(embs)
    idx = FaissIndex(encoder.dim)
    idx.build(embs, ids, texts)
    return idx


def test_build_replug_inputs_shapes_and_alignment(models):
    gen, _, _ = models
    doc_texts = [["doc A", "doc B"], ["doc C", "doc D"]]
    questions = ["What is X?", "Who is Y?"]
    answers = ["the answer", "another"]
    input_ids, attn_mask, prefix_lens, ans_lens = build_replug_inputs(
        doc_texts, questions, answers, gen.tokenizer, max_length=128,
    )
    B, k = 2, 2
    assert input_ids.shape[0] == B * k
    assert attn_mask.shape == input_ids.shape
    assert prefix_lens.shape == (B * k,)
    assert ans_lens.shape == (B * k,)

    # ans_len must be the same across the k axis for each b (so the mixture is well-defined)
    ans_lens_per_b = ans_lens.view(B, k)
    assert torch.equal(ans_lens_per_b[:, 0:1].expand(-1, k), ans_lens_per_b)

    # The token at position prefix_lens[j] should be the FIRST answer token.
    # The chat template's asst opener ends with a newline, so the answer is
    # tokenized without a leading space.
    for j in range(B * k):
        b = j // k
        first_ans_token = gen.tokenizer(answers[b], add_special_tokens=False).input_ids[0]
        assert input_ids[j, prefix_lens[j]].item() == first_ans_token


def test_gather_answer_logprobs_off_by_one():
    """Synthetic logits where for each (j, t) ONE token has logit=0 and all others
    are very negative. After log_softmax, the gathered value should be ≈ 0 ONLY
    if the gather indexes the right (j, pred_pos, target_token) cell.

    A wrong (off-by-one) gather lands on a position where all logits are equal
    → log_softmax = -log(V) ≈ -3.9 → fail the assertion.
    """
    BK, T_max, V = 4, 12, 50
    HUGE_NEG = -1e6
    logits = torch.full((BK, T_max, V), HUGE_NEG)
    prefix_lens = torch.tensor([3, 5, 2, 4])
    ans_lens = torch.tensor([2, 2, 3, 1])
    input_ids = torch.zeros((BK, T_max), dtype=torch.long)
    for j in range(BK):
        p = prefix_lens[j].item()
        a = ans_lens[j].item()
        for t in range(a):
            tok = (j * 7 + t) % V
            input_ids[j, p + t] = tok
            logits[j, p + t - 1, tok] = 0.0  # the lone "real" logit at the right cell

    gathered, _ = gather_answer_logprobs(logits, input_ids, prefix_lens, ans_lens, B=2, k=2)

    flat = gathered.view(BK, 3)
    mask_flat = torch.arange(3).unsqueeze(0) < ans_lens.unsqueeze(-1)
    masked = flat[mask_flat]
    # Correct gather → log_softmax of [0, -HUGE, -HUGE, ...] at target = ≈ 0.
    assert torch.all(masked.abs() < 1e-3), (
        f"gather missed the right cell: values={masked.tolist()}"
    )


def test_gather_handles_short_sequences_in_batch():
    """Regression: when batch elements have varying prefix and ans lengths, the
    longest computed answer_pos can exceed the longest sequence in the batch.
    Without symmetric clamping of pred_pos this raises an out-of-bounds index."""
    BK, T_max, V = 4, 8, 50
    logits = torch.zeros(BK, T_max, V)
    input_ids = torch.zeros(BK, T_max, dtype=torch.long)
    # j=1 has prefix=7 + ans_len=1 → L=8=T_max. T_ans=3 (from j=0). So answer_pos[1, 2] = 9 > T_max-1.
    prefix_lens = torch.tensor([2, 7, 3, 4])
    ans_lens = torch.tensor([3, 1, 2, 1])

    gathered, mask = gather_answer_logprobs(
        logits, input_ids, prefix_lens, ans_lens, B=2, k=2,
    )
    assert gathered.shape == (2, 2, 3)
    assert mask.shape == (2, 3)


def test_replug_step_grads_flow_to_all_three(models):
    """The acceptance test from the plan: after loss.backward(), at least one
    parameter in each of {generator, encoder, projection} has a non-zero grad."""
    gen, enc, proj = models
    gen.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    proj.zero_grad(set_to_none=True)

    idx = _build_toy_index(enc, n=8)

    questions = ["What is the capital of France?", "Who wrote Hamlet?"]
    answers = ["Paris", "Shakespeare"]
    # Route through collate_qa so the chat template path is exercised end-to-end.
    examples = [{"question": q, "answer": a} for q, a in zip(questions, answers, strict=True)]
    batch = collate_qa(examples, gen.tokenizer, max_question_tokens=128)

    loss, metrics = replug_step(
        batch, gen, enc, proj, idx, k=3, tau=0.1, max_length=384,
    )
    assert torch.isfinite(loss)
    loss.backward()

    def _any_nonzero_grad(module: torch.nn.Module) -> bool:
        return any(
            (p.grad is not None) and (p.grad.abs().sum().item() > 0)
            for p in module.parameters()
        )

    assert _any_nonzero_grad(proj), "projection received no gradient"
    assert _any_nonzero_grad(enc), "BGE encoder received no gradient"
    assert _any_nonzero_grad(gen), "Qwen generator received no gradient"
