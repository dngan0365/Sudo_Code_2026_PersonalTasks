"""Command-line entry point for the Vietnamese text processing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.text_processing.pipeline import TextProcessingPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and tokenize the Vietnamese online-news dataset."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/news_dataset.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/news_processed.csv")
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=PROJECT_ROOT / "data/vietnamese-stopwords-dash.txt",
    )
    parser.add_argument(
        "--text-columns", nargs="+", default=["content", "title"],
        help="Columns to clean and tokenize (default: content title).",
    )
    parser.add_argument(
        "--keep-digits", action="store_true", help="Do not remove digits from text."
    )
    parser.add_argument(
        "--no-tokens", action="store_true", help="Only clean text; skip tokenization."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline = TextProcessingPipeline(
        input_path=args.input,
        output_path=args.output,
        stopwords_path=args.stopwords,
        text_columns=args.text_columns,
        remove_digits=not args.keep_digits,
        tokenize=not args.no_tokens,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
