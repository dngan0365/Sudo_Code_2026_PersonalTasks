from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TranslationPair:
    source: str
    target: str


class EVBSgmlDataLoader:
    ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")

    def load(self, corpus_dir: Path, pattern: str, source_lang: str, target_lang: str, max_pairs: int | None = None) -> list[TranslationPair]:
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        pairs: list[TranslationPair] = []
        for file_path in sorted(corpus_dir.glob(pattern)):
            pairs.extend(self._parse_file(file_path, source_lang, target_lang))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs[:max_pairs]
        if not pairs:
            raise ValueError(f"No translation pairs found in {corpus_dir}")
        return pairs

    def _parse_file(self, path: Path, source_lang: str, target_lang: str) -> list[TranslationPair]:
        text = self._read_text(path)
        source_by_number = self._extract_lang_sentences(text, source_lang)
        target_by_number = self._extract_lang_sentences(text, target_lang)
        pairs = []
        for number in sorted(source_by_number):
            if number in target_by_number:
                pairs.append(TranslationPair(source_by_number[number], target_by_number[number]))
        return pairs

    @staticmethod
    def _extract_lang_sentences(text: str, lang: str) -> dict[int, str]:
        pattern = re.compile(rf"<s\s+id=['\"]{re.escape(lang)}(\d+)['\"]>(.*?)</s>", re.IGNORECASE | re.DOTALL)
        return {int(match.group(1)): html.unescape(re.sub(r"\s+", " ", match.group(2)).strip()) for match in pattern.finditer(text)}

    def _read_text(self, path: Path) -> str:
        for encoding in self.ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")
