from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class VietnameseReviewPreprocessor:
    """Clean Vietnamese reviews before feature extraction."""

    tokenize: bool = False
    remove_digits: bool = True

    def clean_text(self, text: str) -> str:
        value = unicodedata.normalize("NFC", html.unescape(str(text))).lower()
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
        value = re.sub(r"<[^>]+>", " ", value)
        if self.remove_digits:
            value = re.sub(r"\d+", " ", value)
        value = "".join(char if self._is_text_char(char) else " " for char in value)
        value = re.sub(r"\s+", " ", value).strip()
        if self.tokenize and value:
            value = self._word_tokenize(value)
        return value

    def transform(self, texts: Iterable[str]) -> pd.Series:
        return pd.Series(texts, dtype="string").fillna("").map(self.clean_text)

    def _is_text_char(self, char: str) -> bool:
        if char.isspace() or char == "_":
            return True
        if self.remove_digits and char.isdigit():
            return False
        return unicodedata.category(char)[0] in {"L", "N"}

    @staticmethod
    def _word_tokenize(text: str) -> str:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError("Install underthesea or run without --tokenize.") from exc
        return word_tokenize(text, format="text")
