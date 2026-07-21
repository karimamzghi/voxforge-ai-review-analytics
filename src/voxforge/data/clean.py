import pandas as pd


COLUMN_MAPPING = {
    "reviews.text": "review_text",
    "reviews.title": "review_title",
    "reviews.rating": "rating",
    "reviews.date": "review_date",
    "reviews.dateAdded": "review_date_added",
    "reviews.dateSeen": "review_date_seen",
    "reviews.didPurchase": "did_purchase",
    "reviews.doRecommend": "do_recommend",
    "reviews.numHelpful": "num_helpful",
    "reviews.username": "username",
    "reviews.userCity": "user_city",
    "reviews.userProvince": "user_province",
    "reviews.id": "review_id",
}


DROP_COLUMNS = [
    "user_city",
    "user_province",
    "did_purchase",
    "review_id",
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_MAPPING).copy()


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]

    return df.drop(columns=columns_to_drop).copy()


def remove_missing_essential_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.dropna(
        subset=["review_text", "rating"]
    ).copy()


def clean_review_text(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["review_text"] = (
        cleaned["review_text"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return cleaned


def clean_review_titles(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["review_title"] = (
        cleaned["review_title"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return cleaned


def validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["rating"] = pd.to_numeric(
        cleaned["rating"],
        errors="coerce",
    )

    cleaned = cleaned[
        cleaned["rating"].between(1, 5)
    ].copy()

    cleaned["rating"] = cleaned["rating"].astype("int8")

    return cleaned


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    date_columns = [
        "review_date",
        "review_date_added",
        "review_date_seen",
        "dateAdded",
        "dateUpdated",
    ]

    for column in date_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(
                cleaned[column],
                errors="coerce",
                utc=True,
            )

    return cleaned


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def remove_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned[
        cleaned["review_text"].str.len() > 0
    ].copy()

    return cleaned


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = rename_columns(df)
    cleaned = drop_unusable_columns(cleaned)
    cleaned = remove_missing_essential_rows(cleaned)
    cleaned = clean_review_text(cleaned)
    cleaned = clean_review_titles(cleaned)
    cleaned = validate_ratings(cleaned)
    cleaned = convert_date_columns(cleaned)
    cleaned = remove_empty_reviews(cleaned)
    cleaned = remove_exact_duplicates(cleaned)

    return cleaned.reset_index(drop=True)
