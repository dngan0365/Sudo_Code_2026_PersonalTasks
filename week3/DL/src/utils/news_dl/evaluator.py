from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report_text: str
    report_dict: dict
    predictions: list[int]


class ModelEvaluator:
    def evaluate(self, model, data_loader, criterion, device: torch.device, label_names: list[str]) -> EvaluationResult:
        model.eval()
        losses: list[float] = []
        predictions: list[int] = []
        labels: list[int] = []
        with torch.no_grad():
            for text, offsets, target in data_loader:
                text = text.to(device)
                offsets = offsets.to(device)
                target = target.to(device)
                logits = model(text, offsets)
                loss = criterion(logits, target)
                losses.append(loss.item())
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                labels.extend(target.cpu().tolist())
        macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
        return EvaluationResult(
            loss=sum(losses) / max(1, len(losses)),
            accuracy=accuracy_score(labels, predictions),
            macro_f1=macro[2],
            weighted_f1=weighted[2],
            report_text=classification_report(labels, predictions, target_names=label_names, zero_division=0),
            report_dict=classification_report(labels, predictions, target_names=label_names, zero_division=0, output_dict=True),
            predictions=predictions,
        )
