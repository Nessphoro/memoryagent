from __future__ import annotations

import torch

from memoryagent.training.muon import (
    ChainedOptimizer,
    Muon,
    partition_params_for_muon,
)


def test_partition_classifies_correctly():
    """2D inner weights → muon; 1D + embedding-named 2D → adam."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 4)   # 2D, embedding
            self.lm_head = torch.nn.Linear(4, 10, bias=False)  # 2D, lm_head
            self.layer = torch.nn.Linear(4, 8)              # 2D, inner — muon
            self.norm = torch.nn.LayerNorm(8)               # 1D weight + 1D bias

    m = Toy()
    muon, adam = partition_params_for_muon(m.named_parameters())
    # Only layer.weight should be muon-bound.
    assert len(muon) == 1
    assert muon[0].shape == (8, 4)
    # adam: layer.bias (1D), norm.weight (1D), norm.bias (1D),
    # embed_tokens.weight (2D embedding), lm_head.weight (2D lm_head)
    assert len(adam) == 5


def test_muon_step_updates_params():
    """Smoke that torch.optim.Muon (re-exported) actually steps a 2D param."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 16))
    opt = Muon([w], lr=0.01)
    w0 = w.detach().clone()
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    assert not torch.equal(w0, w.detach()), "Muon step did not change the param"


def test_chained_optimizer_with_lambda_lr():
    """ChainedOptimizer must satisfy LambdaLR's isinstance + param_groups iteration.

    The LR scheduler reads ``optim.param_groups`` to set ``initial_lr`` and
    later rewrites ``g['lr']`` per group. The chain must round-trip the writes
    through to the underlying optimizers.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))   # muon-shape
    p2 = torch.nn.Parameter(torch.zeros(4))       # adam-shape
    muon = Muon([{"params": [p1], "lr": 0.01}])
    adam = torch.optim.AdamW([{"params": [p2], "lr": 1e-4}])
    chain = ChainedOptimizer([muon, adam])

    # isinstance check (used by LambdaLR.__init__)
    assert isinstance(chain, torch.optim.Optimizer)

    sched = torch.optim.lr_scheduler.LambdaLR(chain, lr_lambda=lambda s: 0.5)
    sched.step()
    assert abs(muon.param_groups[0]["lr"] - 0.005) < 1e-9
    assert abs(adam.param_groups[0]["lr"] - 5e-5) < 1e-9

    # Step optimizer chain end-to-end.
    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    p1_before = p1.detach().clone()
    p2_before = p2.detach().clone()
    chain.step()
    assert not torch.equal(p1_before, p1.detach())
    assert not torch.equal(p2_before, p2.detach())

    # state_dict round-trip.
    sd = chain.state_dict()
    chain2 = ChainedOptimizer([
        Muon([{"params": [torch.nn.Parameter(torch.randn(8, 4))], "lr": 0.01}]),
        torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 1e-4}]),
    ])
    chain2.load_state_dict(sd)
