from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


@dataclass(slots=True)
class VietnameseNewsPreprocessor:
    lowercase: bool = True
    remove_digits: bool = True
    use_underthesea: bool = False

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        if self.use_underthesea:
            try:
                from underthesea import word_tokenize
            except ImportError as exc:
                raise ImportError("Install underthesea or set use_underthesea=false in config.") from exc
            cleaned = word_tokenize(cleaned, format="text")
        return cleaned.split()

    def clean_text(self, text: object) -> str:
        value = html.unescape(str(text) if text is not None else "")
        value = unicodedata.normalize("NFC", value)
        if self.lowercase:
            value = value.lower()
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
        value = re.sub(r"<[^>]+>", " ", value)
        if self.remove_digits:
            value = re.sub(r"\d+", " ", value)
        value = "".join(char if self._is_text_char(char) else " " for char in value)
        return re.sub(r"\s+", " ", value).strip()

    def _is_text_char(self, char: str) -> bool:
        if char.isspace() or char == "_":
            return True
        if self.remove_digits and char.isdigit():
            return False
        return unicodedata.category(char)[0] in {"L", "N"}
