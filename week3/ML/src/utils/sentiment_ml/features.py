from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


@dataclass(slots=True)
class FeatureBuilder:
    """Create TF-IDF features and optional simple numeric features."""

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    include_numeric: bool = False
    min_df: int = 1

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self.scaler = MinMaxScaler() if self.include_numeric else None

    def fit_transform(self, clean_texts: Iterable[str], raw_texts: Iterable[str]):
        text_matrix = self.vectorizer.fit_transform(clean_texts)
        return self._append_numeric(text_matrix, raw_texts, fit=True)

    def transform(self, clean_texts: Iterable[str], raw_texts: Iterable[str]):
        text_matrix = self.vectorizer.transform(clean_texts)
        return self._append_numeric(text_matrix, raw_texts, fit=False)

    def _append_numeric(self, text_matrix, raw_texts: Iterable[str], *, fit: bool):
        if not self.include_numeric:
            return text_matrix
        numeric = self._numeric_features(raw_texts)
        if fit:
            scaled = self.scaler.fit_transform(numeric)
        else:
            scaled = self.scaler.transform(numeric)
        return sparse.hstack([text_matrix, sparse.csr_matrix(scaled)], format="csr")

    @staticmethod
    def _numeric_features(raw_texts: Iterable[str]) -> pd.DataFrame:
        series = pd.Series(raw_texts, dtype="string").fillna("")
        return pd.DataFrame(
            {
                "review_length": series.map(lambda text: len(str(text).split())),
                "sentence_count": series.map(lambda text: max(1, len(re.findall(r"[.!?]+", str(text))) + 1)),
            }
        )
