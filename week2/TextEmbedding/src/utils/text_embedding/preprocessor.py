from __future__ import annotations

import re
import unicodedata

from underthesea import word_tokenize


class VietnameseTextPreprocessor:
    """Normalize and segment Vietnamese text for Word2Vec."""

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        return word_tokenize(cleaned, format="text").split() if cleaned else []
