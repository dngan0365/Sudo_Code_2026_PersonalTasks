from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .data_loader import FolderTextDataLoader
from .dataset import EmbeddingBagCollator, EncodedTextDataset
from .model import SimpleTextClassifier
from .preprocessor import VietnameseNewsPreprocessor
from .trainer import Trainer
from .vocabulary import Vocabulary

LOGGER = logging.getLogger(__name__)


class DeepLearningTextClassificationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> None:
        self._set_seed()
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = FolderTextDataLoader()
        train_data = loader.load(self.config.train_dir, limit_per_class=self.config.limit_per_class)
        test_data = loader.load(
            self.config.test_dir,
            label_names=train_data.label_names,
            limit_per_class=self.config.limit_per_class,
        )
        self._save_dataset_report(train_data, test_data, output_dir / "dataset_report.csv")

        preprocessor = VietnameseNewsPreprocessor(
            lowercase=self.config.lowercase,
            remove_digits=self.config.remove_digits,
            use_underthesea=self.config.use_underthesea,
        )
        LOGGER.info("Tokenizing train and test data")
        train_tokens = [preprocessor.tokenize(text) for text in train_data.texts]
        test_tokens = [preprocessor.tokenize(text) for text in test_data.texts]
        vocab = Vocabulary.build(train_tokens, max_size=self.config.max_vocab_size, min_freq=self.config.min_freq)
        LOGGER.info("Built vocabulary with %s tokens", len(vocab.id_to_token))

        train_dataset = EncodedTextDataset(
            [vocab.encode(tokens, max_tokens=self.config.max_tokens) for tokens in train_tokens],
            train_data.labels,
        )
        test_dataset = EncodedTextDataset(
            [vocab.encode(tokens, max_tokens=self.config.max_tokens) for tokens in test_tokens],
            test_data.labels,
        )
        collator = EmbeddingBagCollator()
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collator)

        device = self._resolve_device()
        model = SimpleTextClassifier(
            vocab_size=len(vocab.id_to_token),
            num_classes=len(train_data.label_names),
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            label_names=train_data.label_names,
            output_dir=output_dir,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            device=device,
        )
        history, best_result = trainer.fit()
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        (output_dir / "classification_report.txt").write_text(best_result.get("classification_report_text", ""), encoding="utf-8")
        metrics = {
            "best_epoch": best_result.get("epoch"),
            "best_metrics": best_result.get("metrics", {}),
            "classification_report": best_result.get("classification_report", {}),
            "label_names": train_data.label_names,
            "vocab_size": len(vocab.id_to_token),
            "config": self._serializable_config(),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_bundle(output_dir / "best_model.pt", vocab, preprocessor, train_data.label_names)
        LOGGER.info("Saved training report and model bundle to %s", output_dir)

    def _save_bundle(self, path: Path, vocab: Vocabulary, preprocessor: VietnameseNewsPreprocessor, label_names: list[str]) -> None:
        torch.save(
            {
                "state_dict": torch.load(self.config.output_dir / "best_model_state.pt", map_location="cpu"),
                "model_config": {
                    "vocab_size": len(vocab.id_to_token),
                    "num_classes": len(label_names),
                    "embedding_dim": self.config.embedding_dim,
                    "hidden_dim": self.config.hidden_dim,
                    "dropout": self.config.dropout,
                },
                "vocabulary": vocab,
                "preprocessor": preprocessor,
                "label_names": label_names,
                "max_tokens": self.config.max_tokens,
            },
            path,
        )
        joblib.dump({"vocabulary": vocab, "label_names": label_names}, self.config.output_dir / "metadata.joblib")

    @staticmethod
    def _save_dataset_report(train_data, test_data, path: Path) -> None:
        rows = []
        for split, data in (("train", train_data), ("test", test_data)):
            for label, name in enumerate(data.label_names):
                rows.append({"split": split, "label": label, "topic": name, "files": data.labels.count(label)})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def _resolve_device(self) -> torch.device:
        requested = self.config.device
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        LOGGER.info("Using device: %s", requested)
        return torch.device(requested)

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _serializable_config(self) -> dict:
        return {
            "train_dir": str(self.config.train_dir),
            "test_dir": str(self.config.test_dir),
            "output_dir": str(self.config.output_dir),
            "limit_per_class": self.config.limit_per_class,
            "max_vocab_size": self.config.max_vocab_size,
            "min_freq": self.config.min_freq,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "dropout": self.config.dropout,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "device": self.config.device,
            "use_underthesea": self.config.use_underthesea,
        }
