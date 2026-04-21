from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from memoryagent.config import TrainConfig
from memoryagent.data.collate import QABatch
from memoryagent.data.loaders import eval_iterator
from memoryagent.device import empty_cache
from memoryagent.eval.retrieval_metrics import retrieval_metrics
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.retrieval.index import FaissIndex
from memoryagent.retrieval.refresh import IndexRefresher
from memoryagent.training.checkpoint import save_checkpoint


def _memory_summary(device: torch.device) -> str:
    if device.type == "cuda":
        used = torch.cuda.memory_allocated(device) / 1e9
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return f"mem={used:.2f}GB peak={peak:.2f}GB"
    if device.type == "mps":
        used = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        return f"mem={used:.2f}GB driver={driver:.2f}GB"
    return ""

StepFn = Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]


def train_loop(
    cfg: TrainConfig,
    *,
    generator: QwenGenerator,
    encoder: BGEEncoder,
    projection: Projection,
    index: FaissIndex,
    refresher: IndexRefresher,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_iter: Iterator[QABatch],
    eval_examples: list[dict] | None,
    step_fn: StepFn,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    all_params = (
        list(generator.parameters())
        + list(encoder.parameters())
        + list(projection.parameters())
    )

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"start training: max_steps={cfg.max_steps} k={cfg.k} tau={cfg.tau} bsz={cfg.batch_size}")
    log(f"output_dir={output_dir} device={device}")

    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        # Refresh index BEFORE the step (so the step sees the freshest snapshot it should).
        # Step indexing: step-1 because the very first refresh fires at step 0.
        if refresher.maybe_refresh(index, step - 1):
            log(f"[step {step}] refreshed index → version {index.version} (size={len(index)})")

        batch = next(train_iter).to(device)
        loss, metrics = step_fn(
            batch, generator, encoder, projection, index,
            k=cfg.k, tau=cfg.tau,
            # max_question_tokens already covers system prompt + envelope +
            # asst opener for the chat template; the extra 32-token slack
            # absorbs BPE boundary effects when the doc is spliced in.
            max_length=cfg.data.max_passage_tokens
            + cfg.data.max_question_tokens
            + cfg.data.max_answer_tokens
            + 32,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.optim.grad_clip)
        optim.step()
        scheduler.step()

        # Help MPS/CUDA allocators recycle between steps. Cheap (just frees the
        # cache pool, doesn't affect held tensors), prevents the slow growth that
        # ate 128GB on M3 last time.
        if step % 5 == 0:
            empty_cache(device)

        if step == 1 or step % cfg.log_every == 0:
            elapsed = time.time() - t0
            sps = step / elapsed
            log(
                f"step {step:6d}/{cfg.max_steps}: "
                f"loss={loss.item():.4f}  "
                f"top1={metrics['scores_top1'].item():.3f}  "
                f"H[w]={metrics['weight_entropy'].item():.3f}  "
                f"({sps:.2f} steps/s)  {_memory_summary(device)}"
            )

        if eval_examples and step % cfg.eval_every == 0:
            it = eval_iterator(
                eval_examples, generator.tokenizer,
                batch_size=cfg.batch_size,
                max_question_tokens=cfg.data.max_question_tokens,
            )
            ev = retrieval_metrics(it, generator, projection, index, k=cfg.k, device=device)
            log(f"[eval @ {step}] {ev}")

        if step % cfg.ckpt_every == 0:
            path = save_checkpoint(
                output_dir,
                step=step,
                generator=generator,
                encoder=encoder,
                projection=projection,
                optim=optim,
                extra={"index_version": index.version},
            )
            log(f"[ckpt] {path.name}")

    # Final
    if eval_examples:
        it = eval_iterator(
            eval_examples, generator.tokenizer,
            batch_size=cfg.batch_size,
            max_question_tokens=cfg.data.max_question_tokens,
        )
        ev = retrieval_metrics(it, generator, projection, index, k=cfg.k, device=device)
        log(f"[final eval] {ev}")
    final_path = save_checkpoint(
        output_dir,
        step=cfg.max_steps,
        generator=generator,
        encoder=encoder,
        projection=projection,
        optim=optim,
        extra={"index_version": index.version, "final": True},
    )
    log(f"[final ckpt] {final_path.name}")
    log(f"done in {time.time() - t0:.1f}s")
