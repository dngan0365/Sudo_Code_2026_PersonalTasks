"""CLI for generating text from a trained LSTM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with a trained character-level LSTM.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "output/best_model.pt")
    parser.add_argument("--prompt", default="Ngay xua")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.text_generation.generator import TextGenerator

    generator = TextGenerator.load(args.model, device=args.device)
    print(generator.generate(args.prompt, args.length, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
