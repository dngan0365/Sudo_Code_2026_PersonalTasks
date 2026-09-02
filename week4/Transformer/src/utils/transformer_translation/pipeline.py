from __future__ import annotations

import json
import logging
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TransformerConfig
from .data_loader import EVBSgmlDataLoader
from .dataset import TranslationCollator, TranslationDataset
from .generator import Translator
from .metrics import corpus_bleu
from .model import TransformerTranslator
from .preprocessor import TextPreprocessor
from .trainer import Trainer
from .vocabulary import Vocabulary

LOGGER = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config

    def run(self) -> None:
        self._set_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        loader = EVBSgmlDataLoader()
        pairs = loader.load(
            self.config.corpus_dir,
            self.config.pattern,
            self.config.source_lang,
            self.config.target_lang,
            self.config.max_pairs,
        )
        random.shuffle(pairs)
        train_pairs, valid_pairs, test_pairs = self._split_pairs(pairs)
        preprocessor = TextPreprocessor(lowercase=self.config.lowercase)
        train_source_tokens = [preprocessor.tokenize(pair.source) for pair in train_pairs]
        train_target_tokens = [preprocessor.tokenize(pair.target) for pair in train_pairs]
        source_vocab = Vocabulary.build(train_source_tokens, self.config.source_max_size, self.config.min_freq)
        target_vocab = Vocabulary.build(train_target_tokens, self.config.target_max_size, self.config.min_freq)
        LOGGER.info("pairs train=%s valid=%s test=%s", len(train_pairs), len(valid_pairs), len(test_pairs))
        LOGGER.info("source_vocab=%s target_vocab=%s", len(source_vocab), len(target_vocab))

        train_loader = self._build_loader(train_pairs, preprocessor, source_vocab, target_vocab, shuffle=True)
        valid_loader = self._build_loader(valid_pairs, preprocessor, source_vocab, target_vocab, shuffle=False)
        device = self._resolve_device()
        model_config = {
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "source_pad_id": source_vocab.pad_id,
            "target_pad_id": target_vocab.pad_id,
            "d_model": self.config.d_model,
            "nhead": self.config.nhead,
            "num_encoder_layers": self.config.num_encoder_layers,
            "num_decoder_layers": self.config.num_decoder_layers,
            "dim_feedforward": self.config.dim_feedforward,
            "dropout": self.config.dropout,
        }
        model = TransformerTranslator(**model_config)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            output_dir=self.config.output_dir,
            model_config=model_config,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            preprocessor=preprocessor,
            max_source_tokens=self.config.max_source_tokens,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            grad_clip=self.config.grad_clip,
            device=device,
        )
        history = trainer.fit()
        pd.DataFrame(history).to_csv(self.config.output_dir / "history.csv", index=False, encoding="utf-8-sig")
        translator = Translator.load(self.config.output_dir / "best_model.pt", device=str(device))
        sample_rows = self._translate_samples(translator, test_pairs)
        pd.DataFrame(sample_rows).to_csv(self.config.output_dir / "translation_samples.csv", index=False, encoding="utf-8-sig")
        (self.config.output_dir / "translation_samples.md").write_text(self._samples_to_markdown(sample_rows), encoding="utf-8")
        bleu = corpus_bleu([row["reference"] for row in sample_rows], [row["prediction"] for row in sample_rows])
        report = {
            "train_pairs": len(train_pairs),
            "valid_pairs": len(valid_pairs),
            "test_pairs": len(test_pairs),
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "best_valid_loss": min(row["valid_loss"] for row in history) if history else None,
            "sample_bleu": bleu,
            "config": self._serializable_config(),
        }
        (self.config.output_dir / "evaluation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved translation quality report to %s", self.config.output_dir)

    def _split_pairs(self, pairs):
        train_end = int(len(pairs) * self.config.train_ratio)
        valid_end = train_end + int(len(pairs) * self.config.valid_ratio)
        return pairs[:train_end], pairs[train_end:valid_end], pairs[valid_end:]

    def _build_loader(self, pairs, preprocessor, source_vocab, target_vocab, shuffle: bool) -> DataLoader:
        sources = [
            source_vocab.encode_source(preprocessor.tokenize(pair.source), self.config.max_source_tokens)
            for pair in pairs
        ]
        targets = [
            target_vocab.encode_target(preprocessor.tokenize(pair.target), self.config.max_target_tokens)
            for pair in pairs
        ]
        dataset = TranslationDataset(sources, targets)
        collator = TranslationCollator(source_vocab.pad_id, target_vocab.pad_id)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collator)

    def _translate_samples(self, translator: Translator, pairs) -> list[dict]:
        rows = []
        for index, pair in enumerate(pairs[: self.config.num_samples]):
            rows.append(
                {
                    "index": index,
                    "source": pair.source,
                    "reference": pair.target,
                    "prediction": translator.translate(pair.source, max_tokens=self.config.generation_max_tokens),
                }
            )
        return rows

    @staticmethod
    def _samples_to_markdown(rows: list[dict]) -> str:
        lines = ["# Translation Samples", ""]
        for row in rows:
            lines.extend(
                [
                    f"## Sample {row['index']}",
                    "",
                    f"Source: {row['source']}",
                    "",
                    f"Reference: {row['reference']}",
                    "",
                    f"Prediction: {row['prediction']}",
                    "",
                ]
            )
        return "\n".join(lines)

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
            "max_pairs": self.config.max_pairs,
            "max_source_tokens": self.config.max_source_tokens,
            "max_target_tokens": self.config.max_target_tokens,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "d_model": self.config.d_model,
            "nhead": self.config.nhead,
            "device": self.config.device,
        }
