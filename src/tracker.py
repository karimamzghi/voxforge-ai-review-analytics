from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def _make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return _make_json_safe(value.tolist())
        except Exception:
            return str(value)
    if hasattr(value, "get_params"):
        try:
            return _make_json_safe(value.get_params())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        return {
            key: _make_json_safe(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def log_experiment(
    model_id=None,
    model_name=None,
    model_family=None,
    features=None,
    preprocessing=None,
    algorithm=None,
    dataset=None,
    training_rows=None,
    validation_rows=None,
    metrics=None,
    training_time_seconds=None,
    inference_time_ms=None,
    artifact_path=None,
    hyperparameters=None,
    output_file=None,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics or {}

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_name": model_name,
        "model_family": model_family,
        "features": features,
        "preprocessing": preprocessing,
        "algorithm": algorithm,
        "dataset": dataset,
        "training_rows": int(training_rows) if training_rows is not None else None,
        "validation_rows": int(validation_rows) if validation_rows is not None else None,
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_precision": metrics.get("weighted_precision"),
        "weighted_recall": metrics.get("weighted_recall"),
        "weighted_f1": metrics.get("weighted_f1"),
        "train_accuracy": metrics.get("train_accuracy"),
        "training_time_seconds": training_time_seconds,
        "inference_time_ms": inference_time_ms,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "hyperparameters": json.dumps(_make_json_safe(hyperparameters), default=str),
    }

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        new_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row])

    new_df.to_csv(output_path, index=False)
    return new_df.tail(1).to_dict(orient="records")[0]
