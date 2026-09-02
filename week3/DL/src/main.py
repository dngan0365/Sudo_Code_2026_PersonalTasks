"""CLI for training a simple neural network text classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple PyTorch neural network for Vietnamese news classification.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-vocab-size", type=int)
    parser.add_argument("--limit-per-class", type=int, help="Limit files per class for a quick smoke test.")
    parser.add_argument("--use-underthesea", action="store_true", help="Use underthesea word_tokenize before training.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.news_dl.config import load_config
    from utils.news_dl.pipeline import DeepLearningTextClassificationPipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config, args, PROJECT_ROOT)
    DeepLearningTextClassificationPipeline(config).run()


if __name__ == "__main__":
    main()
