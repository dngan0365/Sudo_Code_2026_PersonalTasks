from __future__ import annotations

import json
import logging
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import GenerationConfig
from .corpus import TextCorpusLoader
from .dataset import CharSequenceDataset
from .model import CharLSTM
from .trainer import LSTMTrainer
from .vocabulary import CharVocabulary

LOGGER = logging.getLogger(__name__)


class TextGenerationPipeline:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def run(self) -> None:
        self._set_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        corpus_loader = TextCorpusLoader(self.config.lowercase, self.config.normalize_whitespace)
        corpus = corpus_loader.load(
            self.config.corpus_dir,
            pattern=self.config.pattern,
            max_files=self.config.max_files,
            max_chars=self.config.max_chars,
        )
        vocabulary = CharVocabulary.build(corpus.text, min_freq=self.config.min_freq)
        encoded = vocabulary.encode(corpus.text)
        dataset = CharSequenceDataset(encoded, sequence_length=self.config.sequence_length)
        if len(dataset) == 0:
            raise ValueError("Corpus is shorter than sequence_length.")

        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        device = self._resolve_device()
        model_config = {
            "vocab_size": len(vocabulary),
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
        }
        model = CharLSTM(**model_config)
        trainer = LSTMTrainer(
            model=model,
            train_loader=train_loader,
            output_dir=self.config.output_dir,
            vocabulary=vocabulary,
            model_config=model_config,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            grad_clip=self.config.grad_clip,
            device=device,
            prompts=self.config.prompts,
            generation_length=self.config.generation_length,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )
        history, samples = trainer.fit()
        pd.DataFrame(history).to_csv(self.config.output_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(samples).to_csv(self.config.output_dir / "generated_samples.csv", index=False, encoding="utf-8-sig")
        (self.config.output_dir / "generated_samples.md").write_text(self._samples_to_markdown(samples), encoding="utf-8")
        report = {
            "corpus_files": [str(path) for path in corpus.files],
            "corpus_chars": len(corpus.text),
            "vocab_size": len(vocabulary),
            "num_sequences": len(dataset),
            "final_metrics": history[-1] if history else {},
            "config": self._serializable_config(),
        }
        (self.config.output_dir / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved history, generated samples and checkpoint to %s", self.config.output_dir)

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
            "corpus_dir": str(self.config.corpus_dir),
            "pattern": self.config.pattern,
            "max_files": self.config.max_files,
            "max_chars": self.config.max_chars,
            "sequence_length": self.config.sequence_length,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "device": self.config.device,
        }

    @staticmethod
    def _samples_to_markdown(samples: list[dict]) -> str:
        lines = ["# Generated Text Samples", ""]
        for sample in samples:
            lines.append(f"## Epoch {sample['epoch']} - prompt: {sample['prompt']}")
            lines.append("")
            lines.append("```text")
            lines.append(sample["sample"])
            lines.append("```")
            lines.append("")
        return "\n".join(lines)
