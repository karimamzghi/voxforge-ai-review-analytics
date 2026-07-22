"""Central CSV experiment tracking for all sentiment models."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from src.config import MODEL_TRACKING_PATH

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try: return value.item()
        except (ValueError, TypeError): pass
    return str(value)

def log_experiment(
    *, model_id: str, model_name: str, model_family: str, features: str,
    preprocessing: str, algorithm: str, dataset: str, training_rows: int,
    validation_rows: int, metrics: dict[str, Any], training_time_seconds: float,
    inference_time_ms: float, artifact_path: str | Path,
    pretrained_model: str | None = None, epochs: int | None = None,
    batch_size: int | None = None, learning_rate: float | None = None,
    max_length: int | None = None, hyperparameters: dict[str, Any] | None = None,
    output_file: str | Path = MODEL_TRACKING_PATH,
) -> pd.DataFrame:
    """Add or replace one model experiment in the shared tracking CSV."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_id": model_id, "model_name": model_name,
        "model_family": model_family, "features": features,
        "preprocessing": preprocessing, "algorithm": algorithm,
        "pretrained_model": pretrained_model, "dataset": dataset,
        "training_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "train_accuracy": metrics.get("train_accuracy"),
        "validation_accuracy": metrics.get("validation_accuracy", metrics.get("accuracy")),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_ms": float(inference_time_ms),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "max_length": max_length,
        "hyperparameters": json.dumps(_json_safe(hyperparameters or {}), sort_keys=True),
        "artifact_path": str(artifact_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    new_row = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "model_id" not in existing.columns:
            existing["model_id"] = existing.get("model_name", "")
        existing = existing[existing["model_id"].astype(str) != str(model_id)]
        tracking = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        tracking = new_row
    tracking.to_csv(output_path, index=False)
    return tracking
