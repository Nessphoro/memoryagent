"""One-shot: build the initial FAISS index from a corpus and save it to disk.

Useful for large corpora where you want to reuse the index across runs (valid
only before the encoder has been fine-tuned — once it has, the embeddings on
disk no longer match the encoder weights).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memoryagent.config import load_config
from memoryagent.data.loaders import load_dataset
from memoryagent.device import resolve_device
from memoryagent.models.encoder import BGEEncoder
from memoryagent.retrieval.index import FaissIndex
from memoryagent.retrieval.refresh import IndexRefresher


def main(config_path: str, out_path: str | None = None) -> None:
    cfg = load_config(config_path)
    ctx = resolve_device(cfg.device)
    print(f"device: {ctx.device}")

    encoder = BGEEncoder(cfg.model.encoder_name).to(ctx.device)

    print(f"loading dataset: {cfg.data.name}")
    passages, _, _ = load_dataset(cfg.data)
    print(f"  passages: {len(passages):,}")

    refresher = IndexRefresher(encoder, passages, refresh_every=1)
    index = FaissIndex(encoder.dim)
    print("encoding corpus...")
    refresher.refresh(index)
    print(f"  built: size={len(index)} version={index.version}")

    out = Path(out_path or cfg.index.embeddings_path)
    ids_out = Path(str(out).replace(".pt", ".ids.json")) if out.suffix == ".pt" else out.with_suffix(".ids.json")
    index.save(out, ids_out)
    print(f"saved → {out} + {ids_out}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="Override cfg.index.embeddings_path")
    args = p.parse_args()
    main(args.config, out_path=args.out)


if __name__ == "__main__":
    cli()
