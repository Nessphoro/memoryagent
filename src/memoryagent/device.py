from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceCtx:
    device: torch.device
    dtype_generator: torch.dtype
    dtype_encoder: torch.dtype
    dtype_index: torch.dtype
    use_flash_attn: bool
    use_8bit_adam: bool
    use_grad_checkpoint: bool
    pin_memory: bool
    num_workers: int

    @property
    def attn_implementation(self) -> str:
        return "flash_attention_2" if self.use_flash_attn else "sdpa"


def resolve_device(name: str = "auto", *, grad_checkpoint: bool | None = None) -> DeviceCtx:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"

    if name == "cuda":
        return DeviceCtx(
            device=torch.device("cuda"),
            dtype_generator=torch.bfloat16,
            dtype_encoder=torch.float32,
            dtype_index=torch.float16,
            use_flash_attn=True,
            use_8bit_adam=False,
            use_grad_checkpoint=True if grad_checkpoint is None else grad_checkpoint,
            pin_memory=True,
            num_workers=4,
        )
    if name == "mps":
        return DeviceCtx(
            device=torch.device("mps"),
            dtype_generator=torch.bfloat16,
            dtype_encoder=torch.float32,
            dtype_index=torch.float16,
            use_flash_attn=False,
            use_8bit_adam=False,
            use_grad_checkpoint=False if grad_checkpoint is None else grad_checkpoint,
            pin_memory=False,
            num_workers=0,
        )
    if name == "cpu":
        return DeviceCtx(
            device=torch.device("cpu"),
            dtype_generator=torch.float32,
            dtype_encoder=torch.float32,
            dtype_index=torch.float32,
            use_flash_attn=False,
            use_8bit_adam=False,
            use_grad_checkpoint=False if grad_checkpoint is None else grad_checkpoint,
            pin_memory=False,
            num_workers=0,
        )
    raise ValueError(f"Unknown device name: {name!r}")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
