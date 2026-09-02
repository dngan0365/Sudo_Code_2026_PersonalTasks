from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluator import ModelEvaluator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    label_names: list[str]
    output_dir: Path
    epochs: int
    learning_rate: float
    device: torch.device

    def fit(self) -> tuple[list[dict], dict]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        evaluator = ModelEvaluator()
        self.model.to(self.device)
        best_metric = -1.0
        best_result: dict | None = None
        history: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(criterion, optimizer)
            eval_result = evaluator.evaluate(self.model, self.test_loader, criterion, self.device, self.label_names)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": eval_result.loss,
                "test_accuracy": eval_result.accuracy,
                "test_macro_f1": eval_result.macro_f1,
                "test_weighted_f1": eval_result.weighted_f1,
            }
            history.append(row)
            LOGGER.info(
                "epoch=%s train_loss=%.4f test_loss=%.4f accuracy=%.4f macro_f1=%.4f",
                epoch,
                train_loss,
                eval_result.loss,
                eval_result.accuracy,
                eval_result.macro_f1,
            )
            if eval_result.macro_f1 > best_metric:
                best_metric = eval_result.macro_f1
                best_result = {
                    "epoch": epoch,
                    "metrics": row,
                    "classification_report": eval_result.report_dict,
                    "classification_report_text": eval_result.report_text,
                }
                torch.save(self.model.state_dict(), self.output_dir / "best_model_state.pt")
        return history, best_result or {}

    def _train_one_epoch(self, criterion, optimizer) -> float:
        self.model.train()
        losses: list[float] = []
        for text, offsets, labels in self.train_loader:
            text = text.to(self.device)
            offsets = offsets.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(text, offsets)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))
