"""CLI for training Vietnamese Word2Vec embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CBOW and/or Skip-gram on a Vietnamese corpus.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/viwik18.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--architecture", choices=("cbow", "skipgram", "both"), default="both")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--merge-dir", type=Path, help="Merge corpus parts in this directory before training.")
    parser.add_argument("--merge-pattern", default="viwik18_*")
    parser.add_argument("--max-files", type=int, help="Limit merged files for a quick experiment.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.text_embedding.pipeline import EmbeddingPipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    EmbeddingPipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        architecture=args.architecture,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        chunk_size=args.chunk_size,
        merge_dir=args.merge_dir,
        merge_pattern=args.merge_pattern,
        max_files=args.max_files,
    ).run()


if __name__ == "__main__":
    main()
