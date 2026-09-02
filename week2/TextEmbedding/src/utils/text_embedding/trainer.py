from __future__ import annotations

from pathlib import Path

from gensim.models import Word2Vec


class Word2VecTrainer:
    """Configure, train, save and load gensim Word2Vec models."""

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 5, negative: int = 10, epochs: int = 5, workers: int = 2) -> None:
        self.params = dict(vector_size=vector_size, window=window, min_count=min_count, negative=negative, epochs=epochs, workers=workers)

    def train(self, corpus, architecture: str) -> Word2Vec:
        if architecture not in {"cbow", "skipgram"}:
            raise ValueError("architecture must be 'cbow' or 'skipgram'")
        return Word2Vec(sentences=corpus, sg=int(architecture == "skipgram"), **self.params)

    @staticmethod
    def save(model: Word2Vec, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))

    @staticmethod
    def load(path: str | Path) -> Word2Vec:
        return Word2Vec.load(str(path))

    @staticmethod
    def most_similar(model: Word2Vec, word: str, topn: int = 10):
        if word not in model.wv:
            raise KeyError(f"Word not in vocabulary: {word}")
        return model.wv.most_similar(word, topn=topn)
