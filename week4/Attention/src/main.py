"""CLI for training an attention-based text summarizer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LSTM encoder-decoder summarizer with attention.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--valid-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--teacher-forcing-ratio", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.attention_summarization.config import load_config
    from utils.attention_summarization.pipeline import SummarizationPipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config, args, PROJECT_ROOT)
    SummarizationPipeline(config).run()


if __name__ == "__main__":
    main()
