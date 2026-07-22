"""Tests for rating -> sentiment mapping and rating validation."""
import pandas as pd

from src.data.clean import (
    add_sentiment_labels,
    create_sentiment_label,
    validate_ratings,
)


def test_create_sentiment_label_maps_three_classes() -> None:
    assert create_sentiment_label(1) == "negative"
    assert create_sentiment_label(2) == "negative"
    assert create_sentiment_label(3) == "neutral"
    assert create_sentiment_label(4) == "positive"
    assert create_sentiment_label(5) == "positive"


def test_add_sentiment_labels_adds_expected_column() -> None:
    df = pd.DataFrame({"rating": [1, 3, 5]})
    labelled = add_sentiment_labels(df)
    assert list(labelled["sentiment_label"]) == ["negative", "neutral", "positive"]


def test_validate_ratings_drops_out_of_range_values() -> None:
    df = pd.DataFrame({"rating": [0, 1, 3, 5, 6, "x"]})
    cleaned = validate_ratings(df)
    assert sorted(cleaned["rating"].tolist()) == [1, 3, 5]
    assert str(cleaned["rating"].dtype) == "int8"
