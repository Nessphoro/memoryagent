from __future__ import annotations

import torch


def mixture_nll(
    per_token_logp: torch.Tensor,
    log_w: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    """REPLUG per-token mixture NLL in log space.

    For each batch element b and answer position t, the mixture is
        log P_mix(y_t | y_<t, q) = logsumexp_i [ log w_i + log p_i(y_t | ...) ]
    Loss = -mean over masked positions.

    Args:
        per_token_logp: [B, k, T] gathered log p_i(y_t) for the gold answer tokens.
        log_w:          [B, k] log retrieval weights (log_softmax over scores).
        answer_mask:    [B, T] bool — True where there is an answer token to score.

    Returns:
        Scalar loss.
    """
    log_mix = torch.logsumexp(log_w.unsqueeze(-1) + per_token_logp, dim=1)  # [B, T]
    masked = log_mix * answer_mask
    n = answer_mask.sum().clamp_min(1)
    return -(masked.sum() / n)
