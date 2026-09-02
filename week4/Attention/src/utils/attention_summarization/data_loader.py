from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SummaryFrame:
    contents: list[str]
    summaries: list[str]


class ParquetSummaryDataLoader:
    def load(self, path: Path, content_column: str, summary_column: str, limit: int | None = None) -> SummaryFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frame = pd.read_parquet(path, columns=[content_column, summary_column])
        if limit is not None:
            frame = frame.head(limit)
        frame = frame.dropna(subset=[content_column, summary_column])
        frame[content_column] = frame[content_column].astype(str)
        frame[summary_column] = frame[summary_column].astype(str)
        frame = frame[
            frame[content_column].str.strip().astype(bool)
            & frame[summary_column].str.strip().astype(bool)
        ]
        if frame.empty:
            raise ValueError(f"No usable summarization rows found in {path}")
        return SummaryFrame(
            contents=frame[content_column].tolist(),
            summaries=frame[summary_column].tolist(),
        )
