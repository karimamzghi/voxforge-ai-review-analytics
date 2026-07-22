"""Reusable validation helpers for raw, cleaned, and sentiment datasets."""

from collections.abc import Iterable

import pandas as pd

RAW_REQUIRED_COLUMNS = ["reviews.text", "reviews.rating"]
CLEAN_REQUIRED_COLUMNS = ["review_text", "rating", "sentiment_label"]
SENTIMENT_LABELS = {"negative", "neutral", "positive"}


def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raise ValueError when any required column is missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_raw_dataset(df: pd.DataFrame) -> None:
    validate_columns(df, RAW_REQUIRED_COLUMNS)


def validate_clean_dataset(df: pd.DataFrame) -> None:
    validate_columns(df, CLEAN_REQUIRED_COLUMNS)
    invalid = set(df["sentiment_label"].dropna().unique()) - SENTIMENT_LABELS
    if invalid:
        raise ValueError(f"Unexpected sentiment labels: {sorted(invalid)}")


def validate_sentiment_dataset(
    df: pd.DataFrame,
    *,
    text_column: str = "classical_text",
    target_column: str = "sentiment_label",
) -> None:
    """Validate a processed dataset before supervised sentiment training."""
    validate_columns(df, [text_column, target_column])
    invalid = set(df[target_column].dropna().unique()) - SENTIMENT_LABELS
    if invalid:
        raise ValueError(f"Unexpected sentiment labels: {sorted(invalid)}")
