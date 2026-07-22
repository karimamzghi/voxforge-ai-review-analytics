"""Helpers for recording small controlled ablation experiments."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ABLATION_COLUMNS = [
    "experiment_id", "model_name", "changed_factor", "variant", "macro_f1",
    "neutral_f1", "negative_recall", "training_time_seconds", "inference_ms", "notes",
]


def record_ablation(row: dict, output_path: str | Path) -> pd.DataFrame:
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ABLATION_COLUMNS)
    new = pd.DataFrame([{column: row.get(column) for column in ABLATION_COLUMNS}])
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["experiment_id"], keep="last")
    combined.to_csv(path, index=False)
    return combined
