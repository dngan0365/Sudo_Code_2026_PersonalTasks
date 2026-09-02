from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import ReviewDataLoader
from .evaluator import ModelEvaluator
from .features import FeatureBuilder
from .models import ModelFactory
from .preprocessor import VietnameseReviewPreprocessor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SentimentTrainingPipeline:
    input_path: Path
    output_dir: Path
    model_name: str = "logistic"
    has_header: bool = False
    text_column: str = "review"
    label_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    include_numeric: bool = False
    tokenize: bool = False
    remove_digits: bool = True
    class_weight_balanced: bool = False

    label_names: dict[int, str] = None

    def __post_init__(self) -> None:
        if self.label_names is None:
            self.label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        frame = ReviewDataLoader.load(
            self.input_path,
            has_header=self.has_header,
            label_column=self.label_column,
            text_column=self.text_column,
        )
        train_frame, test_frame = train_test_split(
            frame,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=frame[self.label_column],
        )

        preprocessor = VietnameseReviewPreprocessor(tokenize=self.tokenize, remove_digits=self.remove_digits)
        x_train_clean = preprocessor.transform(train_frame[self.text_column])
        x_test_clean = preprocessor.transform(test_frame[self.text_column])

        feature_builder = FeatureBuilder(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            include_numeric=self.include_numeric,
        )
        x_train = feature_builder.fit_transform(x_train_clean, train_frame[self.text_column])
        x_test = feature_builder.transform(x_test_clean, test_frame[self.text_column])

        model_names = list(ModelFactory.SUPPORTED_MODELS) if self.model_name == "all" else [self.model_name]
        factory = ModelFactory(random_state=self.random_state, class_weight_balanced=self.class_weight_balanced)
        evaluator = ModelEvaluator(self.label_names)
        rows: list[dict] = []
        reports: dict[str, dict] = {}
        best_name = None
        best_score = -1.0

        for name in model_names:
            LOGGER.info("Training %s", name)
            model = factory.create(name)
            model.fit(x_train, train_frame[self.label_column])
            predictions = model.predict(x_test)
            scores = evaluator.evaluate(test_frame[self.label_column], predictions)
            row = {"model": name, **{key: value for key, value in scores.items() if not key.startswith("classification_report")}}
            rows.append(row)
            reports[name] = scores["classification_report"]
            LOGGER.info("%s macro_f1=%.4f accuracy=%.4f", name, scores["macro_f1"], scores["accuracy"])

            bundle_path = models_dir / f"{name}.joblib"
            self._save_bundle(bundle_path, model, feature_builder, preprocessor)
            if scores["macro_f1"] > best_score:
                best_name = name
                best_score = scores["macro_f1"]
                self._save_bundle(models_dir / "best_model.joblib", model, feature_builder, preprocessor)

            if len(model_names) == 1:
                self._save_predictions(test_frame, predictions, self.output_dir / "test_predictions.csv")
                (self.output_dir / "classification_report.txt").write_text(scores["classification_report_text"], encoding="utf-8")

        metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        metrics.to_csv(self.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        (self.output_dir / "classification_reports.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved metrics to %s. Best model: %s (macro_f1=%.4f)", self.output_dir, best_name, best_score)

    def _save_bundle(self, path: Path, model: object, feature_builder: FeatureBuilder, preprocessor: VietnameseReviewPreprocessor) -> None:
        joblib.dump(
            {
                "model": model,
                "feature_builder": feature_builder,
                "preprocessor": preprocessor,
                "label_names": self.label_names,
                "text_column": self.text_column,
                "label_column": self.label_column,
            },
            path,
        )

    def _save_predictions(self, test_frame: pd.DataFrame, predictions, path: Path) -> None:
        result = test_frame[[self.label_column, self.text_column]].copy()
        result["prediction"] = predictions
        result["prediction_name"] = [self.label_names.get(int(label), str(label)) for label in predictions]
        result.to_csv(path, index=False, encoding="utf-8-sig")
