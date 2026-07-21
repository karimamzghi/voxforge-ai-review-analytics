import pandas as pd


REQUIRED_COLUMNS = [
    "reviews.text",
    "reviews.rating",
]

# Validate required columns.
def validate_columns(df: pd.DataFrame):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )