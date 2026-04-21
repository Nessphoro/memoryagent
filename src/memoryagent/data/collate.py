from __future__ import annotations

from dataclasses import dataclass

import torch

from memoryagent.prompts import render_query_prompt


@dataclass
class QABatch:
    question_ids: torch.Tensor    # [B, T_q] Qwen-tokenized for encode_query
    question_mask: torch.Tensor   # [B, T_q]
    question_texts: list[str]     # for per-step [doc || question || answer] assembly
    answer_texts: list[str]
    gold_doc_ids: list[str | None] | None = None  # for retrieval eval; optional

    @property
    def size(self) -> int:
        return self.question_ids.size(0)

    def to(self, device: torch.device) -> QABatch:
        return QABatch(
            question_ids=self.question_ids.to(device),
            question_mask=self.question_mask.to(device),
            question_texts=self.question_texts,
            answer_texts=self.answer_texts,
            gold_doc_ids=self.gold_doc_ids,
        )


def collate_qa(
    examples: list[dict],
    tokenizer,
    max_question_tokens: int = 128,
) -> QABatch:
    """Collate a list of {'question': str, 'answer': str, 'gold_doc_id'?: str} into a QABatch.

    Each question is wrapped in Qwen's chat template (system + user + assistant
    opener) so ``encode_query`` sees the same format the model was instruction-
    tuned on. ``max_question_tokens`` budgets the templated length, not the raw
    question — the template adds ~30 tokens of overhead. We do NOT truncate
    here: right-truncation would chop the assistant opener (whose hidden state
    is what we project to BGE space). If a question doesn't fit the budget,
    bump ``max_question_tokens`` rather than letting it silently corrupt the
    format.
    """
    questions = [ex["question"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    gold_doc_ids = [ex.get("gold_doc_id") for ex in examples]

    rendered = [render_query_prompt(tokenizer, q) for q in questions]
    enc = tokenizer(
        rendered,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return QABatch(
        question_ids=enc["input_ids"],
        question_mask=enc["attention_mask"],
        question_texts=questions,
        answer_texts=answers,
        gold_doc_ids=gold_doc_ids,
    )
