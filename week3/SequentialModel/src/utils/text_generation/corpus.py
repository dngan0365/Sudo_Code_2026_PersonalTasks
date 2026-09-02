from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Corpus:
    text: str
    files: list[Path]


class TextCorpusLoader:
    ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")

    def __init__(self, lowercase: bool = False, normalize_whitespace: bool = True) -> None:
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace

    def load(
        self,
        corpus_dir: Path,
        pattern: str = "*.txt",
        max_files: int | None = None,
        max_chars: int | None = None,
    ) -> Corpus:
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        files = sorted(corpus_dir.glob(pattern))
        if max_files is not None:
            files = files[:max_files]

        parts: list[str] = []
        used_files: list[Path] = []
        total_chars = 0
        for file_path in files:
            text = self._normalize(self._read_text(file_path))
            if not text:
                continue
            remaining = None if max_chars is None else max_chars - total_chars
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                text = text[:remaining]
            parts.append(text)
            used_files.append(file_path)
            total_chars += len(text)

        corpus = "\n\n".join(parts)
        if len(corpus) < 1000:
            raise ValueError("Corpus is too small for LSTM training. Increase max_files or max_chars.")
        return Corpus(text=corpus, files=used_files)

    def _read_text(self, path: Path) -> str:
        for encoding in self.ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _normalize(self, text: str) -> str:
        value = unicodedata.normalize("NFC", text)
        if self.lowercase:
            value = value.lower()
        if self.normalize_whitespace:
            value = re.sub(r"[ \t\r\f\v]+", " ", value)
            value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()
