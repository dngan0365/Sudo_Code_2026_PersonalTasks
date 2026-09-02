from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Load tabular news data from a supported file format."""

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported input format: {suffix}")
