from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class EncodedTextDataset(Dataset):
    sequences: list[list[int]]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.sequences[index], self.labels[index]


class EmbeddingBagCollator:
    def __call__(self, batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        offsets = [0]
        for sequence in sequences[:-1]:
            offsets.append(offsets[-1] + len(sequence))
        text = torch.tensor([token_id for sequence in sequences for token_id in sequence], dtype=torch.long)
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return text, offsets_tensor, labels_tensor
