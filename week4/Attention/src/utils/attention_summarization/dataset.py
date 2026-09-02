from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SummarizationDataset(Dataset):
    sources: list[list[int]]
    targets: list[list[int]]

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.sources[index], self.targets[index]


class Seq2SeqCollator:
    def __init__(self, source_pad_id: int, target_pad_id: int) -> None:
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id

    def __call__(self, batch: list[tuple[list[int], list[int]]]):
        sources, targets = zip(*batch)
        source_lengths = torch.tensor([len(item) for item in sources], dtype=torch.long)
        max_source = max(source_lengths).item()
        max_target = max(len(item) for item in targets)
        source_tensor = torch.full((len(batch), max_source), self.source_pad_id, dtype=torch.long)
        target_tensor = torch.full((len(batch), max_target), self.target_pad_id, dtype=torch.long)
        for index, (source, target) in enumerate(zip(sources, targets)):
            source_tensor[index, : len(source)] = torch.tensor(source, dtype=torch.long)
            target_tensor[index, : len(target)] = torch.tensor(target, dtype=torch.long)
        return source_tensor, source_lengths, target_tensor
