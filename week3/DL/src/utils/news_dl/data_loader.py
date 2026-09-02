from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TextClassificationData:
    texts: list[str]
    labels: list[int]
    label_names: list[str]
    paths: list[Path]


class FolderTextDataLoader:
    """Load text files from a class-per-folder dataset."""

    ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")

    def load(self, root_dir: Path, *, label_names: list[str] | None = None, limit_per_class: int | None = None) -> TextClassificationData:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        classes = label_names or sorted(path.name for path in root_dir.iterdir() if path.is_dir())
        texts: list[str] = []
        labels: list[int] = []
        paths: list[Path] = []
        for label, class_name in enumerate(classes):
            class_dir = root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            files = sorted(class_dir.glob("*.txt"))
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for file_path in files:
                texts.append(self._read_text(file_path))
                labels.append(label)
                paths.append(file_path)
        if not texts:
            raise ValueError(f"No .txt files found in {root_dir}")
        return TextClassificationData(texts=texts, labels=labels, label_names=classes, paths=paths)

    def _read_text(self, path: Path) -> str:
        for encoding in self.ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")
