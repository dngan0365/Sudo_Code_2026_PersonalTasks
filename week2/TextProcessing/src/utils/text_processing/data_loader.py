"""Dataset input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class DataLoader:
    """Load and save tabular datasets based on their file extension."""

    READERS = {
        ".json": pd.read_json,
        ".jsonl": lambda path: pd.read_json(path, lines=True),
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
    }

    @classmethod
    def load(cls, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. See README.md for download instructions."
            )
        try:
            reader = cls.READERS[path.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported input format: {path.suffix}") from exc
        return reader(path)

    @staticmethod
    def save(frame: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        elif suffix == ".json":
            frame.to_json(path, orient="records", force_ascii=False, indent=2)
        elif suffix == ".jsonl":
            frame.to_json(path, orient="records", lines=True, force_ascii=False)
        elif suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")
