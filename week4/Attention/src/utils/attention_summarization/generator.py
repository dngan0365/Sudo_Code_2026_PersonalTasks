from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import AttentionSeq2Seq


@dataclass(slots=True)
class SummaryGenerator:
    model: AttentionSeq2Seq
    source_vocab: object
    target_vocab: object
    preprocessor: object
    max_source_tokens: int
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SummaryGenerator":
        resolved_device = torch.device(device)
        bundle = torch.load(path, map_location=resolved_device, weights_only=False)
        model = AttentionSeq2Seq(**bundle["model_config"])
        model.load_state_dict(bundle["state_dict"])
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            source_vocab=bundle["source_vocab"],
            target_vocab=bundle["target_vocab"],
            preprocessor=bundle["preprocessor"],
            max_source_tokens=int(bundle["max_source_tokens"]),
            device=resolved_device,
        )

    def summarize(self, text: str, max_tokens: int = 80) -> str:
        tokens = self.preprocessor.tokenize(text)
        source_ids = self.source_vocab.encode_source(tokens, self.max_source_tokens)
        if not source_ids:
            return ""
        source = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            generated, _ = self.model.generate(source, lengths, max_tokens)
        return self.target_vocab.decode(generated[0])
