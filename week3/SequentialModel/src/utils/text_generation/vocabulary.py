from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(slots=True)
class CharVocabulary:
    char_to_id: dict[str, int]
    id_to_char: list[str]

    @classmethod
    def build(cls, text: str, min_freq: int = 1) -> "CharVocabulary":
        counter = Counter(text)
        chars = sorted(char for char, count in counter.items() if count >= min_freq)
        if not chars:
            raise ValueError("No characters left after applying min_freq.")
        return cls(char_to_id={char: index for index, char in enumerate(chars)}, id_to_char=chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[char] for char in text if char in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[index] for index in ids)

    def __len__(self) -> int:
        return len(self.id_to_char)
