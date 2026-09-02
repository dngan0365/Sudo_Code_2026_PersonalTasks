"""Pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .data_loader import DataLoader
from .preprocessor import VietnameseTextPreprocessor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TextProcessingPipeline:
    input_path: Path
    output_path: Path
    stopwords_path: Path
    text_columns: list[str] = field(default_factory=lambda: ["content", "title"])
    remove_digits: bool = True
    tokenize: bool = True

    def run(self) -> None:
        LOGGER.info("Loading dataset from %s", self.input_path)
        frame = DataLoader.load(self.input_path)
        input_rows = len(frame)
        preprocessor = VietnameseTextPreprocessor(self.stopwords_path, self.remove_digits)
        processed = preprocessor.prepare_frame(frame, self.text_columns, self.tokenize)
        DataLoader.save(processed, self.output_path)
        LOGGER.info("Saved %d/%d rows to %s", len(processed), input_rows, self.output_path)
