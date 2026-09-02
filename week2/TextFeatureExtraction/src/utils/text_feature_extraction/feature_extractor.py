from __future__ import annotations

from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextFeatureExtractor:
    """Fit and persist sparse BoW or TF-IDF text features."""

    def __init__(self, method: str = "tfidf", ngram_range: tuple[int, int] = (1, 2), max_features: int = 5000, min_df: int = 1) -> None:
        if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram range must satisfy 1 <= min <= max")
        vectorizers = {"tfidf": TfidfVectorizer, "bow": CountVectorizer}
        try:
            vectorizer_class = vectorizers[method]
        except KeyError as exc:
            raise ValueError("method must be 'tfidf' or 'bow'") from exc
        self.method = method
        self.vectorizer = vectorizer_class(ngram_range=ngram_range, max_features=max_features, min_df=min_df)

    def fit_transform(self, documents):
        if len(documents) == 0:
            raise ValueError("No valid documents remain after preprocessing")
        return self.vectorizer.fit_transform(documents)

    def save(self, matrix, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(output_dir / f"{self.method}_features.npz", matrix)
        joblib.dump(self.vectorizer, output_dir / f"{self.method}_vectorizer.joblib")
        vocabulary = self.vectorizer.get_feature_names_out()
        (output_dir / f"{self.method}_vocabulary.txt").write_text("\n".join(vocabulary), encoding="utf-8")
