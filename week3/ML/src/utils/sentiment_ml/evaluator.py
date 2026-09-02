from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


@dataclass(slots=True)
class ModelEvaluator:
    """Evaluate classifiers with the metrics from the notebook."""

    label_names: dict[int, str] = field(default_factory=lambda: {0: "Negative", 1: "Neutral", 2: "Positive"})

    def evaluate(self, y_true, y_pred) -> dict:
        labels = list(self.label_names)
        target_names = [self.label_names[label] for label in labels]
        macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": macro[0],
            "macro_recall": macro[1],
            "macro_f1": macro[2],
            "weighted_precision": weighted[0],
            "weighted_recall": weighted[1],
            "weighted_f1": weighted[2],
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            ),
            "classification_report_text": classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0,
            ),
        }
