"""CLI for predicting sentiment with a trained model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Vietnamese review sentiment.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "output/models/best_model.joblib")
    parser.add_argument("--text", action="append", required=True, help="Review text. Repeat this option for many inputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.sentiment_ml.predictor import SentimentPredictor

    predictor = SentimentPredictor.load(args.model)
    for item in predictor.predict(args.text):
        print(f"{item['label']}\t{item['sentiment']}\t{item['text']}")


if __name__ == "__main__":
    main()
