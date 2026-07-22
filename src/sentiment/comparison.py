"""Create consistent benchmark tables and a readable comparison report."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import SENTIMENT_LABELS


def benchmark_from_predictions(
    predictions: pd.DataFrame,
    *,
    model_id: str,
    model_name: str,
    training_time_seconds: float | None = None,
    inference_time_ms: float | None = None,
    model_size_mb: float | None = None,
) -> dict:
    """Calculate one standardized benchmark row from saved predictions."""
    required = {"true_label", "predicted_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Prediction data is missing columns: {sorted(missing)}")

    y_true = predictions["true_label"].astype(str)
    y_pred = predictions["predicted_label"].astype(str)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=SENTIMENT_LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    row = {
        "model_id": model_id,
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_f1": weighted[2],
        "training_time_seconds": training_time_seconds,
        "average_inference_ms": inference_time_ms,
        "model_size_mb": model_size_mb,
        "validation_rows": len(predictions),
    }
    for i, label in enumerate(SENTIMENT_LABELS):
        row[f"{label}_precision"] = precision[i]
        row[f"{label}_recall"] = recall[i]
        row[f"{label}_f1"] = f1[i]
    return row


def path_size_mb(path: str | Path) -> float:
    """Return file or directory size in megabytes."""
    target = Path(path)
    if not target.exists():
        return float("nan")
    size = target.stat().st_size if target.is_file() else sum(
        file.stat().st_size for file in target.rglob("*") if file.is_file()
    )
    return size / (1024 ** 2)


def build_benchmark_table(rows: Iterable[dict], output_path: str | Path | None = None) -> pd.DataFrame:
    benchmark = pd.DataFrame(list(rows))
    if benchmark.empty:
        raise ValueError("At least one benchmark row is required.")
    benchmark = benchmark.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_csv(path, index=False)
    return benchmark


def generate_comparison_report(
    benchmark: pd.DataFrame,
    *,
    selected_model: str,
    reason: str,
    output_path: str | Path,
) -> Path:
    """Write a compact Markdown report from the benchmark table."""
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    display_cols = [
        "model_name", "accuracy", "macro_f1", "negative_f1", "neutral_f1",
        "positive_f1", "average_inference_ms", "model_size_mb",
    ]
    available = [c for c in display_cols if c in benchmark.columns]
    table = benchmark[available].copy()
    numeric = table.select_dtypes(include=[np.number]).columns
    table[numeric] = table[numeric].round(4)
    markdown = table.to_markdown(index=False)
    content = f"""# Sentiment Model Comparison

## Final benchmark

{markdown}

## Selected production model

**{selected_model}**

{reason}

## Decision priorities

Macro F1 is the primary quality metric. Neutral-class F1 and negative recall are checked because these classes are more easily hidden by overall accuracy. Latency and model size are used when transformer quality is close.
"""
    path.write_text(content, encoding="utf-8")
    return path
