from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PipelineConfig:
    train_dir: Path
    test_dir: Path
    output_dir: Path
    limit_per_class: int | None
    lowercase: bool
    remove_digits: bool
    use_underthesea: bool
    max_vocab_size: int
    min_freq: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    max_tokens: int
    seed: int
    device: str


def load_config(config_path: Path, args: Any, project_root: Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = deepcopy(raw)
    _apply_cli_overrides(config, args)
    return PipelineConfig(
        train_dir=_resolve_path(config["data"]["train_dir"], project_root),
        test_dir=_resolve_path(config["data"]["test_dir"], project_root),
        output_dir=_resolve_path(config["output"]["dir"], project_root),
        limit_per_class=config["data"].get("limit_per_class"),
        lowercase=bool(config["preprocessing"]["lowercase"]),
        remove_digits=bool(config["preprocessing"]["remove_digits"]),
        use_underthesea=bool(config["preprocessing"]["use_underthesea"]),
        max_vocab_size=int(config["vocabulary"]["max_size"]),
        min_freq=int(config["vocabulary"]["min_freq"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        batch_size=int(config["training"]["batch_size"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        max_tokens=int(config["training"]["max_tokens"]),
        seed=int(config["training"]["seed"]),
        device=str(config["training"]["device"]),
    )


def _apply_cli_overrides(config: dict[str, Any], args: Any) -> None:
    mapping = {
        "train_dir": ("data", "train_dir"),
        "test_dir": ("data", "test_dir"),
        "output_dir": ("output", "dir"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "learning_rate": ("training", "learning_rate"),
        "max_tokens": ("training", "max_tokens"),
        "max_vocab_size": ("vocabulary", "max_size"),
        "limit_per_class": ("data", "limit_per_class"),
        "device": ("training", "device"),
    }
    for arg_name, path in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[path[0]][path[1]] = value
    if getattr(args, "use_underthesea", False):
        config["preprocessing"]["use_underthesea"] = True


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
