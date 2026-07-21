"""
Model-specific text preprocessing utilities.

This module creates two aligned text representations from the same minimally
cleaned review:

1. transformer_text
   - preserves capitalization and punctuation
   - suitable for pretrained transformer tokenizers

2. classical_text
   - lowercased
   - punctuation and numbers removed
   - tokenized
   - stop words removed while preserving negation
   - lemmatized
   - suitable for TF-IDF or Bag-of-Words models

Normalized duplicate removal is performed on the shared minimally cleaned
combined text before the two branches are created. This keeps both model input
columns row-aligned.
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.voxforge.data.clean import normalize_text_minimally


NLTK_RESOURCES: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


NEGATION_WORDS: set[str] = {
    "no",
    "nor",
    "not",
}


LEMMATIZER = WordNetLemmatizer()


#   Ensure that the NLTK resources required by this module are available.
def ensure_nltk_resources(download: bool = True) -> None:

    missing: list[str] = []

    for resource_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(resource_name)

    if not missing:
        return

    if not download:
        raise LookupError(
            "Missing NLTK resources: " + ", ".join(sorted(missing))
        )

    for resource_name in missing:
        nltk.download(resource_name, quiet=True)


#   Return English stop words while preserving important negation terms. 
def get_sentiment_stop_words() -> set[str]:
   
    ensure_nltk_resources()
    return set(stopwords.words("english")) - NEGATION_WORDS


#   Combine title and review body, then apply shared minimal normalization.
#   The same combined source text is used for both model families so the
#   classical and transformer representations remain directly comparable.
def combine_review_text(
    df: pd.DataFrame,
    title_column: str = "review_title",
    text_column: str = "review_text",
    output_column: str = "normalized_combined_text",
    separator: str = " ",
) -> pd.DataFrame:

    missing_columns = [
        column
        for column in (title_column, text_column)
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required text columns: "
            + ", ".join(sorted(missing_columns))
        )

    processed = df.copy()

    title = processed[title_column].fillna("").astype(str)
    body = processed[text_column].fillna("").astype(str)

    combined = title.str.cat(body, sep=separator)

    processed[output_column] = combined.apply(
        normalize_text_minimally
    )

    return processed


#   Remove duplicate reviews using the shared minimally normalized text.
def remove_normalized_duplicates(
    df: pd.DataFrame,
    text_column: str = "normalized_combined_text",
    rating_column: str = "rating",
    include_rating: bool = True,
) -> pd.DataFrame:
  
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' is required.")

    subset = [text_column]

    if include_rating:
        if rating_column not in df.columns:
            raise KeyError(f"Column '{rating_column}' is required.")
        subset.append(rating_column)

    return (
        df.drop_duplicates(subset=subset, keep="first")
        .reset_index(drop=True)
        .copy()
    )


#   Apply lexical normalization suitable for classical NLP models.
#   punctuation, and isolated one-character tokens are removed.
def clean_text_for_classical_model(text: object) -> str:

    normalized = normalize_text_minimally(text)

    if not normalized:
        return ""

    normalized = normalized.lower()

    # Keep alphabetic English tokens and whitespace.
    normalized = re.sub(r"[^a-z\s]", " ", normalized)

    # Remove isolated one-character tokens.
    normalized = re.sub(r"\b[a-z]\b", " ", normalized)

    # Normalize whitespace after character removal.
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


#   Clean, tokenize, remove stop words, and lemmatize one text value.
def preprocess_classical_text(
    text: object,
    stop_words: set[str] | None = None,
) -> str:

    ensure_nltk_resources()

    cleaned_text = clean_text_for_classical_model(text)

    if not cleaned_text:
        return ""

    if stop_words is None:
        stop_words = get_sentiment_stop_words()

    tokens = word_tokenize(cleaned_text)

    filtered_tokens = [
        token
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    lemmatized_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in filtered_tokens
    ]

    return " ".join(lemmatized_tokens)


def create_model_text_features(
    df: pd.DataFrame,
    title_column: str = "review_title",
    text_column: str = "review_text",
    normalized_column: str = "normalized_combined_text",
    classical_column: str = "classical_text",
    transformer_column: str = "transformer_text",
    remove_duplicates: bool = True,
    include_rating_in_duplicate_key: bool = True,
) -> pd.DataFrame:
 
    processed = combine_review_text(
        df=df,
        title_column=title_column,
        text_column=text_column,
        output_column=normalized_column,
    )

    processed = processed[
        processed[normalized_column].astype("string").str.len().fillna(0) > 0
    ].copy()

    if remove_duplicates:
        processed = remove_normalized_duplicates(
            df=processed,
            text_column=normalized_column,
            include_rating=include_rating_in_duplicate_key,
        )

    # Minimal representation for transformer tokenizers.
    processed[transformer_column] = processed[normalized_column]

    # More aggressive representation for TF-IDF / Bag-of-Words models.
    stop_words = get_sentiment_stop_words()
    processed[classical_column] = processed[normalized_column].apply(
        lambda value: preprocess_classical_text(
            value,
            stop_words=stop_words,
        )
    )

    return processed.reset_index(drop=True)


#   Validate that model-specific text features exist and remain row-aligned.
def validate_model_text_features(
    df: pd.DataFrame,
    normalized_column: str = "normalized_combined_text",
    classical_column: str = "classical_text",
    transformer_column: str = "transformer_text",
) -> None:

    required_columns = {
        normalized_column,
        classical_column,
        transformer_column,
    }
    missing_columns = required_columns.difference(df.columns)

    assert not missing_columns, (
        "Missing model text columns: "
        + ", ".join(sorted(missing_columns))
    )

    assert len(df[classical_column]) == len(df[transformer_column]), (
        "Classical and transformer text columns are not row-aligned."
    )

    assert df[transformer_column].equals(df[normalized_column]), (
        "transformer_text must match normalized_combined_text."
    )

    assert df[normalized_column].astype("string").str.len().gt(0).all(), (
        "Normalized combined text contains empty values."
    )

    assert df[transformer_column].astype("string").str.len().gt(0).all(), (
        "Transformer text contains empty values."
    )


#   Validate and save a feature-engineered dataset as CSV.
def save_feature_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:

    validate_model_text_features(df)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    return path
