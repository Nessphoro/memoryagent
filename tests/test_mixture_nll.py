from __future__ import annotations

import torch

from memoryagent.training.losses import mixture_nll


def test_logsumexp_mixture_matches_naive_in_fp64():
    torch.manual_seed(0)
    B, k, T = 2, 4, 6
    log_w = torch.log_softmax(torch.randn(B, k, dtype=torch.float64), dim=-1)
    log_p = -torch.rand(B, k, T, dtype=torch.float64) * 3  # log-probs in [-3, 0]
    mask = torch.ones(B, T, dtype=torch.bool)

    # Naive: log sum_i w_i * p_i(y_t)
    weighted = log_w.unsqueeze(-1).exp() * log_p.exp()
    naive_log_mix = torch.log(weighted.sum(dim=1))
    naive_loss = -(naive_log_mix * mask).sum() / mask.sum()

    stable_loss = mixture_nll(log_p, log_w, mask)

    assert torch.allclose(naive_loss, stable_loss, atol=1e-12)


def test_mask_excludes_positions():
    torch.manual_seed(0)
    B, k, T = 2, 3, 5
    log_w = torch.log_softmax(torch.randn(B, k, dtype=torch.float64), dim=-1)
    log_p = torch.randn(B, k, T, dtype=torch.float64) - 2

    full_mask = torch.ones(B, T, dtype=torch.bool)
    half_mask = torch.zeros(B, T, dtype=torch.bool)
    half_mask[:, :3] = True

    full = mixture_nll(log_p, log_w, full_mask)
    half = mixture_nll(log_p, log_w, half_mask)

    # Half should equal mean over only the first 3 positions
    log_mix = torch.logsumexp(log_w.unsqueeze(-1) + log_p, dim=1)  # [B, T]
    expected_half = -(log_mix[:, :3].sum() / (B * 3))
    assert torch.allclose(half, expected_half, atol=1e-12)
    assert not torch.allclose(full, half, atol=1e-3)


def test_uniform_weights_recovers_log_mean():
    torch.manual_seed(0)
    B, k, T = 1, 5, 4
    log_w = torch.full((B, k), -float(torch.tensor(k, dtype=torch.float64).log()), dtype=torch.float64)
    log_p = torch.randn(B, k, T, dtype=torch.float64) - 1
    mask = torch.ones(B, T, dtype=torch.bool)

    loss = mixture_nll(log_p, log_w, mask)
    expected = -(torch.log(log_p.exp().mean(dim=1))).mean()
    assert torch.allclose(loss, expected, atol=1e-12)
