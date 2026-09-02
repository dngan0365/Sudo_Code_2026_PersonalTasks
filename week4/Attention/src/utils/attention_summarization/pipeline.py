from __future__ import annotations

import json
import logging
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import AttentionConfig
from .data_loader import ParquetSummaryDataLoader
from .dataset import Seq2SeqCollator, SummarizationDataset
from .generator import SummaryGenerator
from .metrics import average_rouge_l
from .model import AttentionSeq2Seq
from .preprocessor import TextPreprocessor
from .trainer import Trainer
from .vocabulary import Vocabulary

LOGGER = logging.getLogger(__name__)


class SummarizationPipeline:
    def __init__(self, config: AttentionConfig) -> None:
        self.config = config

    def run(self) -> None:
        self._set_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        loader = ParquetSummaryDataLoader()
        train = loader.load(self.config.train_path, self.config.content_column, self.config.summary_column, self.config.train_limit)
        valid = loader.load(self.config.valid_path, self.config.content_column, self.config.summary_column, self.config.valid_limit)
        test = loader.load(self.config.test_path, self.config.content_column, self.config.summary_column, self.config.test_limit)

        preprocessor = TextPreprocessor(lowercase=self.config.lowercase)
        train_source_tokens = [preprocessor.tokenize(text) for text in train.contents]
        train_target_tokens = [preprocessor.tokenize(text) for text in train.summaries]
        source_vocab = Vocabulary.build(train_source_tokens, self.config.source_max_size, self.config.min_freq)
        target_vocab = Vocabulary.build(train_target_tokens, self.config.target_max_size, self.config.min_freq)
        LOGGER.info("source_vocab=%s target_vocab=%s", len(source_vocab), len(target_vocab))

        train_loader = self._build_loader(train, preprocessor, source_vocab, target_vocab, shuffle=True)
        valid_loader = self._build_loader(valid, preprocessor, source_vocab, target_vocab, shuffle=False)

        device = self._resolve_device()
        model_config = {
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "source_pad_id": source_vocab.pad_id,
            "target_pad_id": target_vocab.pad_id,
            "target_sos_id": target_vocab.sos_id,
            "target_eos_id": target_vocab.eos_id,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
        }
        model = AttentionSeq2Seq(**model_config)
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
            teacher_forcing_ratio=self.config.teacher_forcing_ratio,
            grad_clip=self.config.grad_clip,
            device=device,
        )
        history = trainer.fit()
        pd.DataFrame(history).to_csv(self.config.output_dir / "history.csv", index=False, encoding="utf-8-sig")

        generator = SummaryGenerator.load(self.config.output_dir / "best_model.pt", device=str(device))
        sample_rows = self._generate_samples(generator, test)
        sample_frame = pd.DataFrame(sample_rows)
        sample_frame.to_csv(self.config.output_dir / "generated_summaries.csv", index=False, encoding="utf-8-sig")
        (self.config.output_dir / "generated_summaries.md").write_text(self._samples_to_markdown(sample_rows), encoding="utf-8")
        rouge_l = average_rouge_l([row["reference"] for row in sample_rows], [row["prediction"] for row in sample_rows])
        report = {
            "train_rows": len(train.contents),
            "valid_rows": len(valid.contents),
            "test_rows": len(test.contents),
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "best_valid_loss": min(row["valid_loss"] for row in history) if history else None,
            "sample_rouge_l": rouge_l,
            "config": self._serializable_config(),
        }
        (self.config.output_dir / "evaluation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved report and generated summaries to %s", self.config.output_dir)

    def _build_loader(self, data, preprocessor, source_vocab, target_vocab, shuffle: bool) -> DataLoader:
        source_sequences = [
            source_vocab.encode_source(preprocessor.tokenize(text), self.config.max_source_tokens) or [source_vocab.unk_id]
            for text in data.contents
        ]
        target_sequences = [
            target_vocab.encode_target(preprocessor.tokenize(text), self.config.max_target_tokens)
            for text in data.summaries
        ]
        dataset = SummarizationDataset(source_sequences, target_sequences)
        collator = Seq2SeqCollator(source_vocab.pad_id, target_vocab.pad_id)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collator)

    def _generate_samples(self, generator: SummaryGenerator, test) -> list[dict]:
        rows = []
        for index, (content, summary) in enumerate(zip(test.contents, test.summaries)):
            if index >= self.config.num_samples:
                break
            prediction = generator.summarize(content, max_tokens=self.config.max_summary_tokens)
            rows.append(
                {
                    "index": index,
                    "content_preview": content[:500],
                    "reference": summary,
                    "prediction": prediction,
                }
            )
        return rows

    @staticmethod
    def _samples_to_markdown(rows: list[dict]) -> str:
        lines = ["# Generated Summaries", ""]
        for row in rows:
            lines.extend(
                [
                    f"## Sample {row['index']}",
                    "",
                    "### Reference",
                    row["reference"],
                    "",
                    "### Prediction",
                    row["prediction"],
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
            "train_limit": self.config.train_limit,
            "valid_limit": self.config.valid_limit,
            "test_limit": self.config.test_limit,
            "max_source_tokens": self.config.max_source_tokens,
            "max_target_tokens": self.config.max_target_tokens,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "teacher_forcing_ratio": self.config.teacher_forcing_ratio,
            "device": self.config.device,
        }
