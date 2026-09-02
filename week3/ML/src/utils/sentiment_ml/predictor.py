from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib


@dataclass(slots=True)
class SentimentPredictor:
    """Load a saved training bundle and predict review sentiment."""

    model: object
    feature_builder: object
    preprocessor: object
    label_names: dict[int, str]

    @classmethod
    def load(cls, path: Path) -> "SentimentPredictor":
        bundle = joblib.load(path)
        return cls(
            model=bundle["model"],
            feature_builder=bundle["feature_builder"],
            preprocessor=bundle["preprocessor"],
            label_names=bundle.get("label_names", {0: "Negative", 1: "Neutral", 2: "Positive"}),
        )

    def predict(self, texts: Iterable[str]) -> list[dict]:
        raw_texts = list(texts)
        clean_texts = self.preprocessor.transform(raw_texts)
        features = self.feature_builder.transform(clean_texts, raw_texts)
        labels = self.model.predict(features)
        return [
            {
                "text": text,
                "label": int(label),
                "sentiment": self.label_names.get(int(label), str(label)),
            }
            for text, label in zip(raw_texts, labels)
        ]
