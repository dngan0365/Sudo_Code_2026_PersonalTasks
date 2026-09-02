from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @classmethod
    def build(cls, tokenized_texts: Iterable[list[str]], max_size: int, min_freq: int) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        id_to_token = ["<pad>", "<sos>", "<eos>", "<unk>"]
        for token, count in counter.most_common(max(0, max_size - len(id_to_token))):
            if count < min_freq:
                break
            if token not in id_to_token:
                id_to_token.append(token)
        return cls(token_to_id={token: index for index, token in enumerate(id_to_token)}, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.sos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode_source(self, tokens: list[str], max_tokens: int) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_tokens]]

    def encode_target(self, tokens: list[str], max_tokens: int) -> list[int]:
        body = [self.token_to_id.get(token, self.unk_id) for token in tokens[: max_tokens - 2]]
        return [self.sos_id, *body, self.eos_id]

    def decode(self, ids: list[int]) -> str:
        specials = {self.pad_id, self.sos_id, self.eos_id}
        return " ".join(self.id_to_token[index] for index in ids if index not in specials)

    def __len__(self) -> int:
        return len(self.id_to_token)
