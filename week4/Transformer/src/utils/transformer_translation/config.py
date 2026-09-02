from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TransformerConfig:
    corpus_dir: Path
    pattern: str
    source_lang: str
    target_lang: str
    train_ratio: float
    valid_ratio: float
    max_pairs: int | None
    lowercase: bool
    max_source_tokens: int
    max_target_tokens: int
    source_max_size: int
    target_max_size: int
    min_freq: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    grad_clip: float
    seed: int
    device: str
    generation_max_tokens: int
    num_samples: int
    output_dir: Path


def load_config(config_path: Path, args: Any, project_root: Path) -> TransformerConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = deepcopy(raw)
    _apply_cli_overrides(config, args)
    return TransformerConfig(
        corpus_dir=_resolve_path(config["data"]["corpus_dir"], project_root),
        pattern=str(config["data"]["pattern"]),
        source_lang=str(config["data"]["source_lang"]),
        target_lang=str(config["data"]["target_lang"]),
        train_ratio=float(config["data"]["train_ratio"]),
        valid_ratio=float(config["data"]["valid_ratio"]),
        max_pairs=config["data"].get("max_pairs"),
        lowercase=bool(config["preprocessing"]["lowercase"]),
        max_source_tokens=int(config["preprocessing"]["max_source_tokens"]),
        max_target_tokens=int(config["preprocessing"]["max_target_tokens"]),
        source_max_size=int(config["vocabulary"]["source_max_size"]),
        target_max_size=int(config["vocabulary"]["target_max_size"]),
        min_freq=int(config["vocabulary"]["min_freq"]),
        d_model=int(config["model"]["d_model"]),
        nhead=int(config["model"]["nhead"]),
        num_encoder_layers=int(config["model"]["num_encoder_layers"]),
        num_decoder_layers=int(config["model"]["num_decoder_layers"]),
        dim_feedforward=int(config["model"]["dim_feedforward"]),
        dropout=float(config["model"]["dropout"]),
        batch_size=int(config["training"]["batch_size"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        grad_clip=float(config["training"]["grad_clip"]),
        seed=int(config["training"]["seed"]),
        device=str(config["training"]["device"]),
        generation_max_tokens=int(config["generation"]["max_tokens"]),
        num_samples=int(config["generation"]["num_samples"]),
        output_dir=_resolve_path(config["output"]["dir"], project_root),
    )


def _apply_cli_overrides(config: dict[str, Any], args: Any) -> None:
    mapping = {
        "corpus_dir": ("data", "corpus_dir"),
        "output_dir": ("output", "dir"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "max_pairs": ("data", "max_pairs"),
        "max_source_tokens": ("preprocessing", "max_source_tokens"),
        "max_target_tokens": ("preprocessing", "max_target_tokens"),
        "learning_rate": ("training", "learning_rate"),
        "device": ("training", "device"),
    }
    for arg_name, path in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[path[0]][path[1]] = value


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
