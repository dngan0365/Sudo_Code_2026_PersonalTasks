from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    valid_loader: DataLoader
    output_dir: Path
    model_config: dict
    source_vocab: object
    target_vocab: object
    preprocessor: object
    max_source_tokens: int
    epochs: int
    learning_rate: float
    teacher_forcing_ratio: float
    grad_clip: float
    device: torch.device

    def fit(self) -> list[dict]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss(ignore_index=self.target_vocab.pad_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)
        history: list[dict] = []
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, criterion, optimizer)
            valid_loss = self.evaluate_loss(self.valid_loader, criterion)
            row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
            history.append(row)
            LOGGER.info("epoch=%s train_loss=%.4f valid_loss=%.4f", epoch, train_loss, valid_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                self._save_checkpoint(self.output_dir / "best_model.pt")
        return history

    def evaluate_loss(self, data_loader: DataLoader, criterion) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for source, lengths, target in data_loader:
                source = source.to(self.device)
                lengths = lengths.to(self.device)
                target = target.to(self.device)
                logits = self.model(source, lengths, target, teacher_forcing_ratio=0.0)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
                losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    def _run_epoch(self, data_loader: DataLoader, criterion, optimizer) -> float:
        self.model.train()
        losses: list[float] = []
        for source, lengths, target in data_loader:
            source = source.to(self.device)
            lengths = lengths.to(self.device)
            target = target.to(self.device)
            optimizer.zero_grad()
            logits = self.model(source, lengths, target, teacher_forcing_ratio=self.teacher_forcing_ratio)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "source_vocab": self.source_vocab,
                "target_vocab": self.target_vocab,
                "preprocessor": self.preprocessor,
                "max_source_tokens": self.max_source_tokens,
            },
            path,
        )
