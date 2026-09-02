from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import TransformerTranslator


@dataclass(slots=True)
class Translator:
    model: TransformerTranslator
    source_vocab: object
    target_vocab: object
    preprocessor: object
    max_source_tokens: int
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "Translator":
        resolved_device = torch.device(device)
        bundle = torch.load(path, map_location=resolved_device, weights_only=False)
        model = TransformerTranslator(**bundle["model_config"])
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

    def translate(self, text: str, max_tokens: int = 80) -> str:
        source_tokens = self.preprocessor.tokenize(text)
        source_ids = self.source_vocab.encode_source(source_tokens, self.max_source_tokens)
        source = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        generated = [self.target_vocab.sos_id]
        with torch.no_grad():
            memory, source_padding_mask = self.model.encode(source)
            for _ in range(max_tokens):
                target = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits = self.model.decode_step(target, memory, source_padding_mask)
                next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == self.target_vocab.eos_id:
                    break
                generated.append(next_id)
        return self.target_vocab.decode(generated)
