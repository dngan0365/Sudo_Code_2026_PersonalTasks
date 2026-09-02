from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .generator import TextGenerator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LSTMTrainer:
    model: nn.Module
    train_loader: DataLoader
    output_dir: Path
    vocabulary: object
    model_config: dict
    epochs: int
    learning_rate: float
    grad_clip: float
    device: torch.device
    prompts: list[str]
    generation_length: int
    temperature: float
    top_k: int | None

    def fit(self) -> tuple[list[dict], list[dict]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)
        history: list[dict] = []
        samples: list[dict] = []
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(criterion, optimizer)
            perplexity = float(torch.exp(torch.tensor(train_loss)).item())
            row = {"epoch": epoch, "train_loss": train_loss, "perplexity": perplexity}
            history.append(row)
            LOGGER.info("epoch=%s train_loss=%.4f perplexity=%.4f", epoch, train_loss, perplexity)
            samples.extend(self._generate_epoch_samples(epoch))
            if train_loss < best_loss:
                best_loss = train_loss
                self._save_checkpoint(self.output_dir / "best_model.pt")
        return history, samples

    def _train_one_epoch(self, criterion, optimizer) -> float:
        self.model.train()
        losses: list[float] = []
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    def _generate_epoch_samples(self, epoch: int) -> list[dict]:
        generator = TextGenerator(model=self.model, vocabulary=self.vocabulary, device=self.device)
        return [
            {
                "epoch": epoch,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "sample": generator.generate(prompt, self.generation_length, self.temperature, self.top_k),
            }
            for prompt in self.prompts
        ]

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "vocabulary": self.vocabulary,
            },
            path,
        )
