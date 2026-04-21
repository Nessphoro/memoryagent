from __future__ import annotations

from pathlib import Path

import torch

from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    generator: QwenGenerator,
    encoder: BGEEncoder,
    projection: Projection,
    optim: torch.optim.Optimizer,
    extra: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ckpt_step{step:07d}.pt"
    state = {
        "step": step,
        "generator": generator.state_dict(),
        "encoder": encoder.state_dict(),
        "projection": projection.state_dict(),
        "optim": optim.state_dict(),
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    generator: QwenGenerator,
    encoder: BGEEncoder,
    projection: Projection,
    optim: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    generator.load_state_dict(state["generator"])
    encoder.load_state_dict(state["encoder"])
    projection.load_state_dict(state["projection"])
    if optim is not None and "optim" in state:
        optim.load_state_dict(state["optim"])
    return {"step": state["step"], "extra": state.get("extra", {})}
