"""Tests for the transparent production-model selection rule."""
import pandas as pd

from src.sentiment.model_selection import select_production_model


def _benchmark(deberta_f1: float, deberta_neutral_f1: float, deberta_latency: float):
    return pd.DataFrame([
        {"model_id": "distilbert", "model_name": "DistilBERT", "macro_f1": 0.80,
         "neutral_f1": 0.70, "negative_recall": 0.75, "average_inference_ms": 10.0},
        {"model_id": "deberta", "model_name": "DeBERTa-v3-LoRA", "macro_f1": deberta_f1,
         "neutral_f1": deberta_neutral_f1, "negative_recall": 0.75,
         "average_inference_ms": deberta_latency},
    ])


def test_prefers_distilbert_when_gain_is_small() -> None:
    winner, reason = select_production_model(_benchmark(0.805, 0.72, 15.0))
    assert winner["model_name"] == "DistilBERT"
    assert "DistilBERT" in reason


def test_selects_deberta_when_gain_is_meaningful_and_latency_ok() -> None:
    winner, reason = select_production_model(_benchmark(0.86, 0.78, 18.0))
    assert winner["model_name"] == "DeBERTa-v3-LoRA"


def test_keeps_distilbert_when_latency_blows_budget() -> None:
    # Big quality gain, but 5x slower -> latency guard keeps DistilBERT.
    winner, _ = select_production_model(_benchmark(0.90, 0.85, 60.0))
    assert winner["model_name"] == "DistilBERT"
