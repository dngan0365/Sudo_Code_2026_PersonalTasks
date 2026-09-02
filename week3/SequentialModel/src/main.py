"""CLI for training an LSTM text generator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a character-level LSTM for Vietnamese text generation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--generation-length", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.text_generation.config import load_config
    from utils.text_generation.pipeline import TextGenerationPipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config, args, PROJECT_ROOT)
    TextGenerationPipeline(config).run()


if __name__ == "__main__":
    main()
