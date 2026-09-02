from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AttentionConfig:
    train_path: Path
    valid_path: Path
    test_path: Path
    output_dir: Path
    content_column: str
    summary_column: str
    train_limit: int | None
    valid_limit: int | None
    test_limit: int | None
    lowercase: bool
    max_source_tokens: int
    max_target_tokens: int
    source_max_size: int
    target_max_size: int
    min_freq: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    teacher_forcing_ratio: float
    grad_clip: float
    seed: int
    device: str
    max_summary_tokens: int
    num_samples: int


def load_config(config_path: Path, args: Any, project_root: Path) -> AttentionConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = deepcopy(raw)
    _apply_cli_overrides(config, args)
    return AttentionConfig(
        train_path=_resolve_path(config["data"]["train_path"], project_root),
        valid_path=_resolve_path(config["data"]["valid_path"], project_root),
        test_path=_resolve_path(config["data"]["test_path"], project_root),
        output_dir=_resolve_path(config["output"]["dir"], project_root),
        content_column=str(config["data"]["content_column"]),
        summary_column=str(config["data"]["summary_column"]),
        train_limit=config["data"].get("train_limit"),
        valid_limit=config["data"].get("valid_limit"),
        test_limit=config["data"].get("test_limit"),
        lowercase=bool(config["preprocessing"]["lowercase"]),
        max_source_tokens=int(config["preprocessing"]["max_source_tokens"]),
        max_target_tokens=int(config["preprocessing"]["max_target_tokens"]),
        source_max_size=int(config["vocabulary"]["source_max_size"]),
        target_max_size=int(config["vocabulary"]["target_max_size"]),
        min_freq=int(config["vocabulary"]["min_freq"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        batch_size=int(config["training"]["batch_size"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        teacher_forcing_ratio=float(config["training"]["teacher_forcing_ratio"]),
        grad_clip=float(config["training"]["grad_clip"]),
        seed=int(config["training"]["seed"]),
        device=str(config["training"]["device"]),
        max_summary_tokens=int(config["generation"]["max_summary_tokens"]),
        num_samples=int(config["generation"]["num_samples"]),
    )


def _apply_cli_overrides(config: dict[str, Any], args: Any) -> None:
    mapping = {
        "train_path": ("data", "train_path"),
        "valid_path": ("data", "valid_path"),
        "test_path": ("data", "test_path"),
        "output_dir": ("output", "dir"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "train_limit": ("data", "train_limit"),
        "valid_limit": ("data", "valid_limit"),
        "test_limit": ("data", "test_limit"),
        "max_source_tokens": ("preprocessing", "max_source_tokens"),
        "max_target_tokens": ("preprocessing", "max_target_tokens"),
        "learning_rate": ("training", "learning_rate"),
        "teacher_forcing_ratio": ("training", "teacher_forcing_ratio"),
        "device": ("training", "device"),
    }
    for arg_name, path in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[path[0]][path[1]] = value


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
