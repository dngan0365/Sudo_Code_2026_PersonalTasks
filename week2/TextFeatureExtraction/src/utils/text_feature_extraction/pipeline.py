from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .data_loader import DataLoader
from .feature_extractor import TextFeatureExtractor
from .preprocessor import VietnameseNewsPreprocessor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureExtractionPipeline:
    input_path: Path
    output_dir: Path
    stopwords_path: Path
    method: str = "tfidf"
    text_columns: list[str] = field(default_factory=lambda: ["title", "content"])
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 5000
    min_df: int = 1

    def run(self) -> None:
        frame = DataLoader.load(self.input_path)
        prepared, documents = VietnameseNewsPreprocessor(self.stopwords_path).prepare_documents(frame, self.text_columns)
        extractor = TextFeatureExtractor(self.method, self.ngram_range, self.max_features, self.min_df)
        matrix = extractor.fit_transform(documents)
        extractor.save(matrix, self.output_dir)
        metadata = prepared.drop(columns=[column for column in prepared if column.endswith("_processed")])
        metadata.to_csv(self.output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
        LOGGER.info("Saved %s matrix with shape %s to %s", self.method, matrix.shape, self.output_dir)
