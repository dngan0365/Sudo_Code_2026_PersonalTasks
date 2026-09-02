from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .corpus import CorpusMerger, VietnameseWikiCorpus
from .trainer import Word2VecTrainer

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingPipeline:
    input_path: Path
    output_dir: Path
    architecture: str = "both"
    vector_size: int = 100
    window: int = 5
    min_count: int = 5
    negative: int = 10
    epochs: int = 5
    workers: int = 2
    chunk_size: int = 100000
    merge_dir: Path | None = None
    merge_pattern: str = "viwik18_*"
    max_files: int | None = None

    def run(self) -> None:
        if self.merge_dir:
            count = CorpusMerger.merge(self.merge_dir, self.input_path, self.merge_pattern, self.max_files)
            LOGGER.info("Merged %d corpus files into %s", count, self.input_path)
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Corpus not found: {self.input_path}")
        architectures = ("cbow", "skipgram") if self.architecture == "both" else (self.architecture,)
        trainer = Word2VecTrainer(self.vector_size, self.window, self.min_count, self.negative, self.epochs, self.workers)
        for architecture in architectures:
            corpus = VietnameseWikiCorpus(self.input_path, self.chunk_size)
            LOGGER.info("Training %s model", architecture)
            model = trainer.train(corpus, architecture)
            output_path = self.output_dir / f"viwik18_{architecture}.model"
            trainer.save(model, output_path)
            LOGGER.info("Saved model with %d words to %s", len(model.wv), output_path)
