from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from pyvi import ViTokenizer


class VietnameseNewsPreprocessor:
    """Normalize, segment and combine news text before vectorization."""

    def __init__(self, stopwords_path: str | Path) -> None:
        path = Path(stopwords_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stopwords file not found: {path}")
        self.stopwords = {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    @staticmethod
    def clean_text(value: object) -> str:
        if not isinstance(value, str):
            return ""
        text = unicodedata.normalize("NFC", value).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
        return " ".join(text.split())

    def transform_text(self, value: object) -> str:
        segmented = ViTokenizer.tokenize(self.clean_text(value))
        return " ".join(token for token in segmented.split() if token not in self.stopwords)

    def prepare_documents(self, frame: pd.DataFrame, text_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        missing = set(text_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing text columns: {', '.join(sorted(missing))}")
        result = frame.replace(r"^\s*$", pd.NA, regex=True).copy()
        if "content" in text_columns:
            result = result.dropna(subset=["content"]).copy()
        processed_columns = []
        for column in text_columns:
            output_column = f"{column}_processed"
            result[output_column] = result[column].map(self.transform_text)
            processed_columns.append(output_column)
        documents = result[processed_columns].fillna("").agg(" ".join, axis=1).str.strip()
        return result, documents
