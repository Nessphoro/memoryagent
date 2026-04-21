from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    generator_name: str = "Qwen/Qwen3.5-0.8B"
    encoder_name: str = "BAAI/bge-small-en-v1.5"
    encoder_dim: int = 384
    generator_hidden_dim: int = 1024
    use_query_instruction: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


class IndexConfig(BaseModel):
    embeddings_path: Path = Path("embeddings_v0.pt")
    ids_path: Path = Path("ids.json")
    refresh_every: int = 500


class DataConfig(BaseModel):
    name: Literal["toy", "nq"] = "toy"
    train_size: int | None = None
    eval_size: int | None = None
    max_question_tokens: int = 128   # chat template eats ~30 tokens of overhead
    max_answer_tokens: int = 32
    max_passage_tokens: int = 192
    corpus_subsample: int | None = None


class OptimConfig(BaseModel):
    optimizer: Literal["adamw", "muon"] = "adamw"
    # Baseline (AdamW) per-module LRs. When optimizer == "muon", these still
    # govern the AdamW group (1D params + embeddings + lm_head); the Muon
    # group (matrix weights) uses ``base_lr * muon_lr_multiplier``.
    lr_generator: float = 1e-5
    lr_encoder: float = 2e-5
    lr_projection: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    use_8bit_adam: bool = False
    # Muon knobs (ignored when optimizer == "adamw"). Defaults assume the
    # built-in torch.optim.Muon (PyTorch >= 2.11) with adjust_lr_fn set to
    # "match_rms_adamw" — this is Moonshot's RMS-matching adjustment that
    # lets the matrix-param Muon group reuse the AdamW LRs above directly,
    # so no separate lr_multiplier is needed.
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["match_rms_adamw", "original"] = "match_rms_adamw"


class TrainConfig(BaseModel):
    seed: int = 42
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    grad_checkpoint: bool | None = None
    batch_size: int = 2
    k: int = 5
    tau: float = 0.1
    max_steps: int = 1000
    eval_every: int = 100
    log_every: int = 10
    ckpt_every: int = 500
    kl_to_ref: float = 0.0
    output_dir: Path = Path("runs/default")
    model: ModelConfig = Field(default_factory=ModelConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)


def load_config(path: str | Path) -> TrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return TrainConfig.model_validate(data)
