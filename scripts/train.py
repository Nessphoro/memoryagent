"""Main training entrypoint. Wire YAML config → train_loop with replug_step."""

from __future__ import annotations

import argparse
from pathlib import Path

from memoryagent.config import load_config
from memoryagent.data.loaders import infinite_train_iterator, load_dataset
from memoryagent.device import resolve_device
from memoryagent.models.encoder import BGEEncoder
from memoryagent.models.generator import QwenGenerator
from memoryagent.models.projection import Projection
from memoryagent.retrieval.index import FaissIndex
from memoryagent.retrieval.refresh import IndexRefresher
from memoryagent.seed import seed_all
from memoryagent.training.checkpoint import load_checkpoint
from memoryagent.training.loop import train_loop
from memoryagent.training.optim import build_optimizer, build_scheduler
from memoryagent.training.replug import replug_step


def main(config_path: str, *, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    seed_all(cfg.seed)
    ctx = resolve_device(cfg.device, grad_checkpoint=cfg.grad_checkpoint)
    print(f"device: {ctx.device}, gen dtype: {ctx.dtype_generator}, ckpt: {ctx.use_grad_checkpoint}")

    print(f"loading models: {cfg.model.generator_name}, {cfg.model.encoder_name}")
    generator = QwenGenerator(
        cfg.model.generator_name,
        dtype=ctx.dtype_generator,
        attn_implementation=ctx.attn_implementation,
    ).to(ctx.device)
    encoder = BGEEncoder(
        cfg.model.encoder_name,
        use_query_instruction=cfg.model.use_query_instruction,
        query_instruction=cfg.model.query_instruction,
    ).to(ctx.device)
    projection = Projection(generator.hidden_size, encoder.dim).to(ctx.device)

    if ctx.use_grad_checkpoint:
        generator.model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled on generator")

    print(f"loading dataset: {cfg.data.name}")
    passages, train_qa, eval_qa = load_dataset(cfg.data)
    print(f"  passages={len(passages)} train={len(train_qa)} eval={len(eval_qa)}")

    refresher = IndexRefresher(
        encoder, passages, refresh_every=cfg.index.refresh_every,
    )
    index = FaissIndex(encoder.dim)
    # The first iteration of train_loop will trigger the initial build via
    # maybe_refresh (last_refresh_step is None), so we don't build it here.
    print("(initial index will build on step 1)")

    optim = build_optimizer(generator, encoder, projection, cfg.optim, ctx)
    scheduler = build_scheduler(optim, cfg.optim, max_steps=cfg.max_steps)

    if resume:
        meta = load_checkpoint(
            Path(resume),
            generator=generator, encoder=encoder, projection=projection,
            optim=optim, map_location=ctx.device,
        )
        print(f"resumed from {resume} (step={meta['step']})")

    train_iter = infinite_train_iterator(
        train_qa, generator.tokenizer,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        max_question_tokens=cfg.data.max_question_tokens,
    )

    train_loop(
        cfg,
        generator=generator,
        encoder=encoder,
        projection=projection,
        index=index,
        refresher=refresher,
        optim=optim,
        scheduler=scheduler,
        train_iter=train_iter,
        eval_examples=eval_qa,
        step_fn=replug_step,
        device=ctx.device,
        output_dir=cfg.output_dir,
    )


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="path to a checkpoint .pt to resume from")
    args = p.parse_args()
    main(args.config, resume=args.resume)


if __name__ == "__main__":
    cli()
