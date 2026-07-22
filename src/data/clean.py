"""
General, model-agnostic data cleaning utilities.

This module is responsible for structural and data-quality cleaning that should
be applied before model specific text preprocessing.

Typical responsibilities:
- standardize column names
- remove unusable columns
- handle essential missing values
- normalize review title and review body safely
- validate ratings
- convert date columns
- remove empty reviews
- remove exact duplicate rows

"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


COLUMN_MAPPING: dict[str, str] = {
    "name": "product_name",
    "brand": "brand",
    "categories": "categories",
    "primaryCategories": "primary_categories",
    "reviews.title": "review_title",
    "reviews.text": "review_text",
    "reviews.rating": "rating",
    "reviews.date": "review_date",

}

DEFAULT_DROP_COLUMNS: tuple[str, ...] = (
    "reviews.username",
    "reviews.userCity",       # 100% null
    "reviews.userProvince",   # 100% null
    "reviews.didPurchase",    # 10 non-null
    "reviews.id",             # 71 non-null
    "id",                     # 71 non-null
    "keys",                   # barcode / external-key junk
    "manufacturer",           # redundant with `brand` (both "Amazon")
    "reviews.sourceURLs",     # scraper URLs
    "sourceURLs",             # duplicate of above, one source only
    "reviews.dateAdded",      # scraper timestamp, 63% missing
    "dateAdded",              # duplicate, one source only
    "dateUpdated",            # scraper metadata, one source
    "reviews.dateSeen",       # concatenated scrape timestamps
    "imageURLs",              # scraper metadata (keep only for UI thumbnails)
    "manufacturerNumber",     # not needed
    "reviews.doRecommend",
    "reviews.numHelpful",
)


DATE_COLUMNS: tuple[str, ...] = (
    "review_date",
    "review_date_added",
    "review_date_seen",
    "dateAdded",
    "dateUpdated",
)

#   Rename raw dataset columns to consistent project-friendly names.
#   Existing columns that are not listed in COLUMN_MAPPING are preserved.
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    return df.rename(columns=COLUMN_MAPPING).copy()


# Drop columns that are not useful for downstream modeling.
# Missing columns are ignored so the function remains reusable across
def drop_unusable_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:

    columns_to_drop = [column for column in columns if column in df.columns]
    return df.drop(columns=columns_to_drop).copy()


#    Remove rows that are missing fields required for supervised sentiment work.
def remove_missing_essential_rows(
    df: pd.DataFrame,
    essential_columns: Iterable[str] = ("review_text", "rating"),
) -> pd.DataFrame:

    required = list(essential_columns)
    missing_columns = [column for column in required if column not in df.columns]

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return df.dropna(subset=required).copy()


#   This function intentionally preserves capitalization, punctuation,
#   sentence order, and negation so the result remains suitable for both
#   classical and transformer preprocessing branches.
def normalize_text_minimally(value: object) -> str:
  
    if pd.isna(value):
        return ""

    text = str(value)

    # Decode HTML entities such as &amp; and &#39;.
    text = html.unescape(text)

    # Remove script and style blocks.
    text = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove HTML comments and remaining tags.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Remove control and non-printable characters while preserving whitespace.
    text = "".join(
        character
        for character in text
        if character.isprintable() or character.isspace()
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

#    Minimally clean the review body while preserving natural language signals.
def clean_review_text(df: pd.DataFrame) -> pd.DataFrame:

    if "review_text" not in df.columns:
        raise KeyError("Column 'review_text' is required.")

    cleaned = df.copy()
    cleaned["review_text"] = cleaned["review_text"].apply(
        normalize_text_minimally
    )
    return cleaned

#   Minimally clean review titles.
def clean_review_titles(df: pd.DataFrame) -> pd.DataFrame:

    cleaned = df.copy()

    if "review_title" not in cleaned.columns:
        cleaned["review_title"] = ""

    cleaned["review_title"] = cleaned["review_title"].apply(
        normalize_text_minimally
    )
    return cleaned

#   Convert ratings to numeric values and remove values outside the valid range.
def validate_ratings(
    df: pd.DataFrame,
    minimum: int = 1,
    maximum: int = 5,
) -> pd.DataFrame:
 
    if "rating" not in df.columns:
        raise KeyError("Column 'rating' is required.")

    cleaned = df.copy()

    cleaned["rating"] = pd.to_numeric(
        cleaned["rating"],
        errors="coerce",
    )

    cleaned = cleaned[
        cleaned["rating"].between(minimum, maximum, inclusive="both")
    ].copy()

    cleaned["rating"] = cleaned["rating"].astype("int8")

    return cleaned


#   Convert available date columns to pandas datetimes.
def convert_date_columns(
    df: pd.DataFrame,
    date_columns: Iterable[str] = DATE_COLUMNS,
) -> pd.DataFrame:

    cleaned = df.copy()

    for column in date_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(
                cleaned[column],
                errors="coerce",
                utc=True,
            )

    return cleaned


# Convert Amazon star ratings into three sentiment classes.
def create_sentiment_label(rating: int) -> str:

    if rating <= 2:
        return "negative"

    if rating == 3:
        return "neutral"

    return "positive"

# Add sentiment labels derived from Amazon star ratings.
def add_sentiment_labels(
    df: pd.DataFrame,
    rating_column: str = "rating",
    output_column: str = "sentiment_label",
) -> pd.DataFrame:

    if rating_column not in df.columns:
        raise KeyError(f"Column '{rating_column}' is required.")

    cleaned = df.copy()

    cleaned[output_column] = cleaned[rating_column].apply(
        create_sentiment_label
    )

    return cleaned


#   Remove rows whose review body is empty after minimal normalization.
def remove_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
 
    if "review_text" not in df.columns:
        raise KeyError("Column 'review_text' is required.")

    cleaned = df.copy()

    cleaned = cleaned[
        cleaned["review_text"].astype("string").str.len().fillna(0) > 0
    ].copy()

    return cleaned


#   Remove rows that are exact duplicates across all columns.
def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    return df.drop_duplicates().copy()


#   Raise an AssertionError when core post-cleaning expectations are violated.
def validate_clean_dataset(df: pd.DataFrame) -> None:

    required_columns = {"review_text", "review_title", "rating"}
    missing_columns = required_columns.difference(df.columns)
    expected_labels = {"negative", "neutral", "positive"}

    assert not missing_columns, (
        "Missing required cleaned columns: "
        + ", ".join(sorted(missing_columns))
    )
    assert df["review_text"].notna().all(), (
        "review_text still contains missing values."
    )
    assert df["review_text"].astype("string").str.len().gt(0).all(), (
        "review_text still contains empty strings."
    )
    assert df["rating"].between(1, 5, inclusive="both").all(), (
        "rating contains values outside the expected range 1-5."
    )
    assert df.duplicated().sum() == 0, (
        "Exact duplicate rows remain in the cleaned dataset."
    )
    assert set(df["sentiment_label"].unique()) <= expected_labels


#   Run the complete model-agnostic cleaning pipeline.
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    cleaned = rename_columns(df)
    cleaned = drop_unusable_columns(cleaned)
    cleaned = remove_missing_essential_rows(cleaned)
    cleaned = clean_review_text(cleaned)
    cleaned = clean_review_titles(cleaned)
    cleaned = validate_ratings(cleaned)
    cleaned = add_sentiment_labels(cleaned)
    cleaned = convert_date_columns(cleaned)
    cleaned = remove_empty_reviews(cleaned)
    cleaned = remove_exact_duplicates(cleaned)
    cleaned = cleaned.reset_index(drop=True)

    validate_clean_dataset(cleaned)

    return cleaned
