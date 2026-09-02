from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReviewDataLoader:
    """Load and validate review sentiment datasets."""

    @staticmethod
    def load(
        path: Path,
        *,
        has_header: bool = False,
        label_column: str = "label",
        text_column: str = "review",
    ) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            if has_header:
                frame = pd.read_csv(path)
            else:
                frame = pd.read_csv(path, header=None, names=[label_column, text_column])
        elif suffix == ".json":
            frame = pd.read_json(path)
        elif suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input format: {suffix}")

        missing = {label_column, text_column}.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        prepared = frame[[label_column, text_column]].copy()
        prepared = prepared.dropna(subset=[label_column, text_column])
        prepared[label_column] = prepared[label_column].astype(int)
        prepared[text_column] = prepared[text_column].astype(str)
        prepared = prepared[prepared[text_column].str.strip().astype(bool)].reset_index(drop=True)
        if prepared.empty:
            raise ValueError("Dataset is empty after removing missing reviews and labels.")
        return prepared
