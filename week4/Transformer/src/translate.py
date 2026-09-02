"""CLI for translating text with a trained Transformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate English text to Vietnamese.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "output/best_model.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.transformer_translation.generator import Translator

    translator = Translator.load(args.model, device=args.device)
    print(translator.translate(args.text, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
