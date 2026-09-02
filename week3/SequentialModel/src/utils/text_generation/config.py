from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class GenerationConfig:
    corpus_dir: Path
    pattern: str
    max_files: int | None
    max_chars: int | None
    lowercase: bool
    normalize_whitespace: bool
    min_freq: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    grad_clip: float
    seed: int
    device: str
    prompts: list[str]
    generation_length: int
    temperature: float
    top_k: int | None
    output_dir: Path


def load_config(config_path: Path, args: Any, project_root: Path) -> GenerationConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = deepcopy(raw)
    _apply_cli_overrides(config, args)
    return GenerationConfig(
        corpus_dir=_resolve_path(config["data"]["corpus_dir"], project_root),
        pattern=str(config["data"]["pattern"]),
        max_files=config["data"].get("max_files"),
        max_chars=config["data"].get("max_chars"),
        lowercase=bool(config["preprocessing"]["lowercase"]),
        normalize_whitespace=bool(config["preprocessing"]["normalize_whitespace"]),
        min_freq=int(config["vocabulary"]["min_freq"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        sequence_length=int(config["training"]["sequence_length"]),
        batch_size=int(config["training"]["batch_size"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        grad_clip=float(config["training"]["grad_clip"]),
        seed=int(config["training"]["seed"]),
        device=str(config["training"]["device"]),
        prompts=list(config["generation"]["prompts"]),
        generation_length=int(config["generation"]["length"]),
        temperature=float(config["generation"]["temperature"]),
        top_k=config["generation"].get("top_k"),
        output_dir=_resolve_path(config["output"]["dir"], project_root),
    )


def _apply_cli_overrides(config: dict[str, Any], args: Any) -> None:
    mapping = {
        "corpus_dir": ("data", "corpus_dir"),
        "output_dir": ("output", "dir"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "sequence_length": ("training", "sequence_length"),
        "learning_rate": ("training", "learning_rate"),
        "max_files": ("data", "max_files"),
        "max_chars": ("data", "max_chars"),
        "generation_length": ("generation", "length"),
        "temperature": ("generation", "temperature"),
        "top_k": ("generation", "top_k"),
        "device": ("training", "device"),
    }
    for arg_name, path in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[path[0]][path[1]] = value


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
