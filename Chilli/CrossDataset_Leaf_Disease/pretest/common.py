from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(cfg["_project_root"]) / p).resolve()


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["runtime"].get("device", "auto")).lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda, but CUDA is not available.")

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("runtime.device=mps, but MPS is not available.")

    return torch.device(requested)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False


def configure_runtime(cfg: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(
            cfg["runtime"].get("cudnn_benchmark", True)
        ) and not bool(cfg["runtime"].get("deterministic", False))


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(x: Any):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        raise TypeError(f"Not JSON serializable: {type(x)}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def canonical_label_order(cfg: dict[str, Any], labels: list[str] | set[str]) -> list[str]:
    present = set(labels)
    return [
        x for x in cfg["data"]["canonical_labels"]
        if x in present
    ]
