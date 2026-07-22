"""CSV-based experiment tracking for model comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import EXPERIMENT_TRACKING_PATH

# Convert common ML and scientific Python objects into JSON-safe values.
def _json_serializer(value: Any) -> Any:

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    if hasattr(value, "get_params"):
        return {
            "estimator": value.__class__.__name__,
            "parameters": value.get_params(deep=False),
        }

    return str(value)

#  Append one experiment row to the central tracking CSV
def log_experiment(
    *,
    model_name: str,
    dataset: str,
    training_rows: int,
    validation_rows: int,
    metrics: dict[str, float],
    training_time_seconds: float,
    inference_time_ms: float,
    artifact_path: str | Path,
    hyperparameters: dict[str, Any] | None = None,
    output_file: str | Path = EXPERIMENT_TRACKING_PATH,
) -> pd.DataFrame:

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_hyperparameters = json.dumps(
        hyperparameters or {},
        default=_json_serializer,
        sort_keys=True,
    )

    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "model_name": model_name,
                "dataset": dataset,
                "training_rows": int(training_rows),
                "validation_rows": int(validation_rows),
                "accuracy": metrics.get("accuracy"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "training_time_seconds": float(training_time_seconds),
                "inference_time_ms": float(inference_time_ms),
                "hyperparameters": serialized_hyperparameters,
                "artifact_path": str(artifact_path),
            }
        ]
    )

    if output_path.exists():
        existing = pd.read_csv(output_path)
        result = pd.concat([existing, row], ignore_index=True)
    else:
        result = row

    result.to_csv(output_path, index=False)

    return row