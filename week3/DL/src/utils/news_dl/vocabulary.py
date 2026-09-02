from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"

    @classmethod
    def build(cls, tokenized_texts: Iterable[list[str]], *, max_size: int = 30000, min_freq: int = 2) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        id_to_token = ["<pad>", "<unk>"]
        for token, count in counter.most_common(max(0, max_size - len(id_to_token))):
            if count < min_freq:
                break
            id_to_token.append(token)
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, tokens: list[str], *, max_tokens: int) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_tokens]]
        return ids or [self.unk_id]
