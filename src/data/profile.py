from typing import Any

import pandas as pd


#   Generate a high-level overview of the dataset.
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:

    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    summary = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate Rows": df.duplicated().sum(),
        "Memory Usage (MB)": round(memory_mb, 2),
        "Total Missing Values": int(df.isna().sum().sum()),
    }

    return pd.DataFrame(
        summary.items(),
        columns=["Metric", "Value"],
    )

#   Compute descriptive statistics for review text.
def text_statistics(
    df: pd.DataFrame,
    text_column: str = "reviews.text",
) -> pd.DataFrame:

    text = (
        df[text_column]
        .fillna("")
        .astype(str)
    )

    char_count = text.str.len()

    word_count = text.str.split().str.len()

    stats = {
        "Average Characters": round(char_count.mean(), 2),
        "Median Characters": round(char_count.median(), 2),
        "Minimum Characters": int(char_count.min()),
        "Maximum Characters": int(char_count.max()),
        "Average Words": round(word_count.mean(), 2),
        "Median Words": round(word_count.median(), 2),
        "Minimum Words": int(word_count.min()),
        "Maximum Words": int(word_count.max()),
    }

    return pd.DataFrame(
        stats.items(),
        columns=["Metric", "Value"],
    )

# calculate vocabulary size
def vocabulary_size(
    df: pd.DataFrame,
    text_column: str = "reviews.text",
) -> int:

    words = (
        df[text_column]
        .fillna("")
        .str.lower()
        .str.split()
        .explode()
    )

    return words.nunique()

from typing import Any

# Generate data quality report
def data_quality_report(
    df: pd.DataFrame,
) -> dict[str, Any]:

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": (
            df.isna()
            .sum()
            .to_dict()
        ),
        "data_types": (
            df.dtypes.astype(str)
            .to_dict()
        ),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum()
            / 1024**2,
            2,
        ),
    }

    return report

# Generate the 3 data profiles
def generate_profile(
    df: pd.DataFrame,
) -> dict:

    return {
        "overview": dataset_overview(df),
        "text_statistics": text_statistics(df),
        "quality_report": data_quality_report(df),
    }