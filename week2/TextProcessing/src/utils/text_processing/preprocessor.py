"""Vietnamese news cleaning and tokenization."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


class VietnameseTextPreprocessor:
    """Clean dataset fields and tokenize Vietnamese text using underthesea."""

    REQUIRED_TEXT_COLUMN = "content"
    FILL_UNKNOWN_COLUMNS = ("author", "source", "topic", "url")

    def __init__(self, stopwords_path: str | Path, remove_digits: bool = True) -> None:
        self.stopwords = self._load_stopwords(stopwords_path)
        self.remove_digits = remove_digits

    @staticmethod
    def _load_stopwords(path: str | Path) -> set[str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Stopwords file not found: {path}")
        with path.open(encoding="utf-8") as file:
            # The supplied list uses underscores for multi-syllable words.
            return {line.strip().lower() for line in file if line.strip()}

    def clean_text(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        text = unicodedata.normalize("NFC", value).lower()
        if self.remove_digits:
            text = re.sub(r"\d+", " ", text)
        text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
        return " ".join(text.split())

    def tokenize(self, text: str) -> list[str]:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError("Install underthesea or run with --no-tokens.") from exc
        segmented = word_tokenize(text, format="text")
        return [token for token in segmented.split() if token not in self.stopwords]

    @staticmethod
    def _source_from_url(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        host = urlparse(value).netloc.lower().removeprefix("www.")
        return host or None

    def prepare_frame(
        self,
        frame: pd.DataFrame,
        text_columns: list[str],
        tokenize: bool = True,
    ) -> pd.DataFrame:
        missing = set(text_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing text columns: {', '.join(sorted(missing))}")
        if self.REQUIRED_TEXT_COLUMN not in frame.columns:
            raise ValueError("Dataset must contain a 'content' column")

        result = frame.copy()
        # Treat empty strings as missing without accidentally copying content into title.
        result = result.replace(r"^\s*$", pd.NA, regex=True)
        result = result.dropna(subset=[self.REQUIRED_TEXT_COLUMN]).copy()
        if "processed" in result.columns:
            result = result.drop(columns=["processed"])

        if "source" not in result.columns and "url" in result.columns:
            result["source"] = pd.NA
        if "source" in result.columns and "url" in result.columns:
            inferred = result["url"].map(self._source_from_url)
            result["source"] = result["source"].fillna(inferred)

        for column in self.FILL_UNKNOWN_COLUMNS:
            if column in result.columns:
                result[column] = result[column].fillna("Unknown").astype("string")
        if "crawled_at" in result.columns:
            result["crawled_at"] = pd.to_datetime(result["crawled_at"], errors="coerce")

        for column in text_columns:
            cleaned_column = f"{column}_clean"
            result[cleaned_column] = result[column].map(self.clean_text)
            if tokenize:
                # Store a space-separated value so CSV/JSON output remains portable.
                result[f"{column}_tokens"] = result[cleaned_column].map(
                    lambda text: " ".join(self.tokenize(text))
                )

        result["content_len"] = result["content"].astype("string").str.len()
        return result
