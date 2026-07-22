"""Load the selected sentiment model and predict new or batch reviews."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import ID_TO_LABEL, SENTIMENT_LABELS


class SentimentPredictor:
    def __init__(self, model_path: str | Path, *, batch_size: int = 32, max_length: int = 256):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.kind = "sklearn" if self.model_path.suffix == ".joblib" else "transformer"
        if self.kind == "sklearn":
            self.model = joblib.load(self.model_path)
            self.tokenizer = None
            self.device = "cpu"
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device); self.model.eval()

    def predict_batch(self, texts: Iterable[str]) -> pd.DataFrame:
        values = pd.Series(list(texts), dtype="string").fillna("").astype(str)
        if self.kind == "sklearn":
            labels = self.model.predict(values)
            probabilities = self.model.predict_proba(values)
            classes = [str(x) for x in self.model.classes_]
        else:
            all_probs = []
            for start in range(0, len(values), self.batch_size):
                batch = values.iloc[start:start + self.batch_size].tolist()
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    probs = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
            probabilities = np.vstack(all_probs) if all_probs else np.empty((0, len(SENTIMENT_LABELS)))
            classes = [ID_TO_LABEL[i] for i in range(len(SENTIMENT_LABELS))]
            labels = np.array([classes[i] for i in probabilities.argmax(axis=1)]) if len(probabilities) else np.array([])
        result = pd.DataFrame({"predicted_sentiment": labels})
        for index, label in enumerate(classes):
            result[f"probability_{label}"] = probabilities[:, index]
        probability_cols = [c for c in result if c.startswith("probability_")]
        result["sentiment_confidence"] = result[probability_cols].max(axis=1)
        return result

    def predict_one(self, text: str) -> dict:
        row = self.predict_batch([text]).iloc[0]
        return {
            "label": row["predicted_sentiment"],
            "confidence": float(row["sentiment_confidence"]),
            "probabilities": {label: float(row.get(f"probability_{label}", 0.0)) for label in SENTIMENT_LABELS},
        }


def load_selected_predictor(selection_path: str | Path) -> SentimentPredictor:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    return SentimentPredictor(payload["artifact_path"])


def enrich_reviews(
    reviews: pd.DataFrame,
    predictor: SentimentPredictor,
    *,
    text_column: str = "transformer_text",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if text_column not in reviews:
        raise KeyError(f"Column '{text_column}' was not found.")
    predictions = predictor.predict_batch(reviews[text_column])
    enriched = pd.concat([reviews.reset_index(drop=True), predictions], axis=1)
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(path, index=False)
    return enriched
