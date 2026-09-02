"""CLI for summarizing text with a trained attention model."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a summary with a trained attention summarizer.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "output/best_model.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.attention_summarization.generator import SummaryGenerator

    generator = SummaryGenerator.load(args.model, device=args.device)
    print(generator.summarize(args.text, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
