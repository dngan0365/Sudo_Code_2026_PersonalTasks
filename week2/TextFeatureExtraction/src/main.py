"""CLI for Vietnamese text feature extraction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract BoW or TF-IDF features from Vietnamese news.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/news_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--stopwords", type=Path, default=PROJECT_ROOT / "data/vietnamese-stopwords-dash.txt")
    parser.add_argument("--method", choices=("tfidf", "bow"), default="tfidf")
    parser.add_argument("--text-columns", nargs="+", default=["title", "content"])
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from utils.text_feature_extraction.pipeline import FeatureExtractionPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    FeatureExtractionPipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        stopwords_path=args.stopwords,
        method=args.method,
        text_columns=args.text_columns,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
    ).run()


if __name__ == "__main__":
    main()
