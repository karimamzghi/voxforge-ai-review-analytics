"""Simple, reusable sentiment error analysis."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ["probability_negative", "probability_neutral", "probability_positive"]


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "true_label", "predicted_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Prediction data is missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["is_correct"] = frame["true_label"] == frame["predicted_label"]
    frame["error_type"] = frame["true_label"].astype(str) + " -> " + frame["predicted_label"].astype(str)
    frame["text_length"] = frame["text"].str.len()
    frame["word_count"] = frame["text"].str.split().str.len()
    frame["contains_contrast"] = frame["text"].str.contains(
        r"\b(but|however|although|though|yet|except)\b", case=False, regex=True
    )
    available_probs = [c for c in PROBABILITY_COLUMNS if c in frame.columns]
    frame["confidence"] = frame[available_probs].max(axis=1) if available_probs else np.nan
    return frame


def save_error_subsets(analysis: pd.DataFrame, output_dir: str | Path, model_id: str) -> dict[str, Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    files = {
        "all_errors": output / f"{model_id}_errors.csv",
        "high_confidence_errors": output / f"{model_id}_high_confidence_errors.csv",
        "contrast_errors": output / f"{model_id}_contrast_errors.csv",
    }
    errors = analysis.loc[~analysis["is_correct"]].sort_values("confidence", ascending=False)
    errors.to_csv(files["all_errors"], index=False)
    errors.loc[errors["confidence"].fillna(0) >= 0.80].to_csv(files["high_confidence_errors"], index=False)
    errors.loc[errors["contains_contrast"]].to_csv(files["contrast_errors"], index=False)
    return files


def compare_two_models(first: pd.DataFrame, second: pd.DataFrame, *, first_name: str, second_name: str) -> dict[str, pd.DataFrame]:
    """Compare models row-by-row. Both files must contain the same validation rows."""
    left = first.reset_index(drop=True).copy(); right = second.reset_index(drop=True).copy()
    if len(left) != len(right):
        raise ValueError("Prediction files have different row counts.")
    if not left["true_label"].astype(str).equals(right["true_label"].astype(str)):
        raise ValueError("Prediction files do not use the same row order or labels.")
    if "text" in left and "text" in right and not left["text"].astype(str).equals(right["text"].astype(str)):
        raise ValueError("Prediction files do not contain the same review order.")
    merged = pd.DataFrame({
        "text": left.get("text", pd.Series(range(len(left)))),
        "true_label": left["true_label"],
        f"{first_name}_prediction": left["predicted_label"],
        f"{second_name}_prediction": right["predicted_label"],
    })
    first_ok = merged[f"{first_name}_prediction"] == merged["true_label"]
    second_ok = merged[f"{second_name}_prediction"] == merged["true_label"]
    return {
        "shared_errors": merged.loc[~first_ok & ~second_ok],
        f"{first_name}_wins": merged.loc[first_ok & ~second_ok],
        f"{second_name}_wins": merged.loc[~first_ok & second_ok],
        "disagreements": merged.loc[merged[f"{first_name}_prediction"] != merged[f"{second_name}_prediction"]],
    }
