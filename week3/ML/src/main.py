"""CLI for Vietnamese sentiment classification with classical ML models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Vietnamese review sentiment classifiers.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument(
        "--model",
        choices=("logistic", "svm-linear", "svm-rbf", "svm-poly", "multinomial-nb", "bernoulli-nb", "all"),
        default="logistic",
    )
    parser.add_argument("--has-header", action="store_true", help="Read the CSV header from the input file.")
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--include-numeric", action="store_true", help="Add review length and sentence count features.")
    parser.add_argument("--tokenize", action="store_true", help="Use underthesea word_tokenize before vectorizing.")
    parser.add_argument("--keep-digits", action="store_true")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use balanced class weights where supported.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.sentiment_ml.pipeline import SentimentTrainingPipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    SentimentTrainingPipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        has_header=args.has_header,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        include_numeric=args.include_numeric,
        tokenize=args.tokenize,
        remove_digits=not args.keep_digits,
        class_weight_balanced=args.class_weight_balanced,
    ).run()


if __name__ == "__main__":
    main()
