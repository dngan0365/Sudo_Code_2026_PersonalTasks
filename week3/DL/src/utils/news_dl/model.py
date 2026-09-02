from __future__ import annotations

import torch
from torch import nn


class SimpleTextClassifier(nn.Module):
    """EmbeddingBag text classifier for bag-of-words style neural classification."""

    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        return self.classifier(embedded)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
