from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class CharSequenceDataset(Dataset):
    encoded_text: list[int]
    sequence_length: int

    def __len__(self) -> int:
        return max(0, len(self.encoded_text) - self.sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.encoded_text[index : index + self.sequence_length]
        target = self.encoded_text[index + 1 : index + self.sequence_length + 1]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)
