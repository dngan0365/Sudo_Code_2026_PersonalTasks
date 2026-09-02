from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


@dataclass(slots=True)
class TextPreprocessor:
    lowercase: bool = True

    def tokenize(self, text: object) -> list[str]:
        value = html.unescape(str(text) if text is not None else "")
        value = unicodedata.normalize("NFC", value)
        if self.lowercase:
            value = value.lower()
        value = re.sub(r"\s+", " ", value).strip()
        return value.split()
