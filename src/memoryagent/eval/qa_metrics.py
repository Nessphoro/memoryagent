"""Exact-match and token-F1 for short-answer QA, SQuAD-style.

NQ-open answers are multi-valued (several acceptable phrasings per question);
``em`` and ``f1`` take the max over the gold list. The text normalizer is the
SQuAD canon: lowercase, strip articles, strip punctuation, collapse whitespace.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def em(pred: str, golds: list[str] | str) -> float:
    """1.0 iff normalized pred matches any normalized gold."""
    if isinstance(golds, str):
        golds = [golds]
    p = _normalize(pred)
    return float(any(_normalize(g) == p for g in golds))


def _f1_one(pred_toks: list[str], gold_toks: list[str]) -> float:
    if not pred_toks or not gold_toks:
        # SQuAD convention: 1.0 if both empty (no answer expected, none given), else 0.0.
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def f1(pred: str, golds: list[str] | str) -> float:
    """Token-overlap F1, max over the gold list."""
    if isinstance(golds, str):
        golds = [golds]
    p_toks = _normalize(pred).split()
    return max(_f1_one(p_toks, _normalize(g).split()) for g in golds)


def aggregate(preds: list[str], gold_lists: list[list[str]]) -> dict[str, float]:
    """Mean EM and F1 over a batch of (pred, gold_list) pairs."""
    if not preds:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    ems = [em(p, g) for p, g in zip(preds, gold_lists, strict=True)]
    f1s = [f1(p, g) for p, g in zip(preds, gold_lists, strict=True)]
    return {
        "em": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
        "n": len(preds),
    }
