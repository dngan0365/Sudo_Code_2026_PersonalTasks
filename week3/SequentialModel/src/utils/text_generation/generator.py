from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import CharLSTM


@dataclass(slots=True)
class TextGenerator:
    model: CharLSTM
    vocabulary: object
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TextGenerator":
        resolved_device = torch.device(device)
        bundle = torch.load(path, map_location=resolved_device, weights_only=False)
        model = CharLSTM(**bundle["model_config"])
        model.load_state_dict(bundle["state_dict"])
        model.to(resolved_device)
        model.eval()
        return cls(model=model, vocabulary=bundle["vocabulary"], device=resolved_device)

    def generate(self, prompt: str, length: int = 300, temperature: float = 0.8, top_k: int | None = 40) -> str:
        if not prompt:
            prompt = self.vocabulary.id_to_char[0]
        encoded = self.vocabulary.encode(prompt)
        if not encoded:
            encoded = [0]

        generated = list(encoded)
        hidden = None
        current = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, hidden = self.model(current, hidden)
            next_input = current[:, -1:]
            for _ in range(length):
                logits, hidden = self.model(next_input, hidden)
                next_id = self._sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
                generated.append(next_id)
                next_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        return self.vocabulary.decode(generated)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return int(indices.gather(-1, sampled).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
