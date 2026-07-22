"""Dataset preparation and splitting helpers for sentiment models."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.data.validate import validate_sentiment_dataset


def prepare_classical_sentiment_data(
    df: pd.DataFrame,
    *,
    text_column: str = "classical_text",
    target_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Return valid, non-empty text and labels for classical NLP models."""
    validate_sentiment_dataset(
        df,
        text_column=text_column,
        target_column=target_column,
    )

    model_df = df[[text_column, target_column]].dropna().copy()
    model_df[text_column] = model_df[text_column].astype(str).str.strip()
    model_df = model_df[model_df[text_column].str.len() > 0]
    return model_df.reset_index(drop=True)


def split_sentiment_data(
    model_df: pd.DataFrame,
    *,
    text_column: str = "classical_text",
    target_column: str = "sentiment_label",
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Create stratified train, validation, and test splits.

    ``validation_size`` is the fraction taken from the remaining train/validation
    partition after the test split. With both values at 0.20, the final shares
    are 64% train, 16% validation, and 20% test.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    X = model_df[text_column]
    y = model_df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
