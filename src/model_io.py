"""Reusable helpers for saving and loading project artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def _prepare_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a fitted model or sklearn Pipeline with joblib."""
    output_path = _prepare_parent(path)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path) -> Any:
    """Load a model previously saved with :func:`save_model`."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    return joblib.load(input_path)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save metrics as formatted JSON."""
    output_path = _prepare_parent(path)
    serializable = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)
    return output_path


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Save a DataFrame as CSV."""
    output_path = _prepare_parent(path)
    dataframe.to_csv(output_path, index=index)
    return output_path
