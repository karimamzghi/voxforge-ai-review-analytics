"""Transparent production model selection."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def select_production_model(
    benchmark: pd.DataFrame,
    *,
    distilbert_name: str = "DistilBERT",
    deberta_name_contains: str = "DeBERTa",
    minimum_f1_gain: float = 0.01,
    maximum_latency_multiplier: float = 2.0,
) -> tuple[pd.Series, str]:
    """Select the best model, preferring DistilBERT when transformer quality is close."""
    if benchmark.empty:
        raise ValueError("Benchmark table is empty.")
    distil = benchmark.loc[benchmark["model_name"].str.fullmatch(distilbert_name, case=False, na=False)]
    deberta = benchmark.loc[benchmark["model_name"].str.contains(deberta_name_contains, case=False, na=False)]
    if distil.empty or deberta.empty:
        winner = benchmark.sort_values("macro_f1", ascending=False).iloc[0]
        return winner, "Selected the highest Macro F1 because both transformer candidates were not available."

    d = distil.iloc[0]; b = deberta.iloc[0]
    gain = float(b["macro_f1"] - d["macro_f1"])
    d_latency = float(d.get("average_inference_ms", float("nan")))
    b_latency = float(b.get("average_inference_ms", float("nan")))
    latency_ratio = b_latency / d_latency if d_latency > 0 and pd.notna(b_latency) else 1.0
    neutral_gain = float(b.get("neutral_f1", 0) - d.get("neutral_f1", 0))
    negative_recall_gain = float(b.get("negative_recall", 0) - d.get("negative_recall", 0))

    meaningful_quality_gain = gain >= minimum_f1_gain and (neutral_gain > 0 or negative_recall_gain > 0)
    acceptable_latency = latency_ratio <= maximum_latency_multiplier
    if meaningful_quality_gain and acceptable_latency:
        reason = (
            f"DeBERTa improved Macro F1 by {gain:.4f}, improved a priority class, "
            f"and its latency multiplier ({latency_ratio:.2f}x) stayed within the limit."
        )
        return b, reason
    reason = (
        f"DistilBERT was selected because DeBERTa's Macro F1 gain was {gain:.4f} "
        f"and its latency multiplier was {latency_ratio:.2f}x; the quality gain did not justify extra complexity."
    )
    return d, reason


def save_selection(winner: pd.Series, reason: str, *, artifact_path: str | Path, output_path: str | Path) -> Path:
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": str(winner["model_id"]),
        "model_name": str(winner["model_name"]),
        "artifact_path": str(artifact_path),
        "macro_f1": float(winner["macro_f1"]),
        "reason": reason,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
