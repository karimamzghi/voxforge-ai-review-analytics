"""Tests for stratified train/validation/test splitting."""
import pandas as pd

from src.sentiment.data import split_sentiment_data


def test_split_proportions_and_stratification() -> None:
    frame = pd.DataFrame({
        "classical_text": [f"review number {i}" for i in range(300)],
        "sentiment_label": (["positive"] * 150) + (["neutral"] * 90) + (["negative"] * 60),
    })
    X_train, X_val, X_test, y_train, y_val, y_test = split_sentiment_data(frame)

    assert len(X_train) + len(X_val) + len(X_test) == 300
    # 64 / 16 / 20 split
    assert len(X_test) == 60
    assert abs(len(X_train) - 192) <= 1
    assert abs(len(X_val) - 48) <= 1
    # every class present in every split (stratified)
    for split in (y_train, y_val, y_test):
        assert set(split.unique()) == {"positive", "neutral", "negative"}
