"""CLI for predicting news topic with a trained PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Vietnamese news topic from text.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "output/best_model.pt")
    parser.add_argument("--text", action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.news_dl.predictor import NewsTopicPredictor

    predictor = NewsTopicPredictor.load(args.model)
    for result in predictor.predict(args.text):
        print(f"{result['label']}\t{result['topic']}\t{result['confidence']:.4f}\t{result['text']}")


if __name__ == "__main__":
    main()
