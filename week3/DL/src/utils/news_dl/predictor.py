from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .model import SimpleTextClassifier


@dataclass(slots=True)
class NewsTopicPredictor:
    model: SimpleTextClassifier
    vocabulary: object
    preprocessor: object
    label_names: list[str]
    max_tokens: int
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "NewsTopicPredictor":
        resolved_device = torch.device(device)
        bundle = torch.load(path, map_location=resolved_device, weights_only=False)
        model = SimpleTextClassifier(**bundle["model_config"])
        model.load_state_dict(bundle["state_dict"])
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            vocabulary=bundle["vocabulary"],
            preprocessor=bundle["preprocessor"],
            label_names=bundle["label_names"],
            max_tokens=int(bundle["max_tokens"]),
            device=resolved_device,
        )

    def predict(self, texts: Iterable[str]) -> list[dict]:
        raw_texts = list(texts)
        results = []
        with torch.no_grad():
            for text in raw_texts:
                tokens = self.preprocessor.tokenize(text)
                ids = self.vocabulary.encode(tokens, max_tokens=self.max_tokens)
                text_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
                offsets = torch.tensor([0], dtype=torch.long, device=self.device)
                probabilities = torch.softmax(self.model(text_tensor, offsets), dim=1)[0]
                label = int(torch.argmax(probabilities).item())
                results.append(
                    {
                        "text": text,
                        "label": label,
                        "topic": self.label_names[label],
                        "confidence": float(probabilities[label].item()),
                    }
                )
        return results
