from __future__ import annotations

from pathlib import Path

from .preprocessor import VietnameseTextPreprocessor


class CorpusMerger:
    """Merge extracted Wikipedia corpus parts into one UTF-8 training file."""

    @staticmethod
    def merge(source_dir: str | Path, output_path: str | Path, pattern: str = "viwik18_*", max_files: int | None = None) -> int:
        source_dir, output_path = Path(source_dir), Path(output_path)
        files = sorted(path for path in source_dir.glob(pattern) if path.is_file())
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"No corpus files matching {pattern!r} in {source_dir}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as destination:
            for path in files:
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                destination.write(b"\n")
        return len(files)


class VietnameseWikiCorpus:
    """Re-iterable, memory-efficient corpus consumed by gensim."""

    def __init__(self, path: str | Path, chunk_size: int = 100000, min_tokens: int = 4) -> None:
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.min_tokens = min_tokens
        self.preprocessor = VietnameseTextPreprocessor()

    def __iter__(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"Corpus not found: {self.path}")
        with self.path.open(encoding="utf-8", errors="ignore") as file:
            remainder = ""
            while block := file.read(self.chunk_size):
                text = remainder + block
                split_at = text.rfind(" ")
                if split_at < 0:
                    remainder = text
                    continue
                text, remainder = text[:split_at], text[split_at + 1 :]
                tokens = self.preprocessor.tokenize(text)
                if len(tokens) >= self.min_tokens:
                    yield tokens
            tokens = self.preprocessor.tokenize(remainder)
            if len(tokens) >= self.min_tokens:
                yield tokens
