"""Data access helpers for the VoxForge insights pipeline.

This repository is the source of truth where the insights layer reads the
processed analytics dataset. Summary, recommendation, report, and exporter
modules should depend on this class instead of calling ``pandas.read_csv``
directly.

- reads the clustered review CSV;
- validates the columns needed by the insights layer;
- normalises topic and sentiment values;
- exposes reusable filtered views and lightweight aggregates.

It does not run sentiment inference or clustering. Those steps remain in the
existing ``src.sentiment`` and ``src.clustering`` packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.config import CLUSTERED_REVIEWS_PATH, SENTIMENT_LABELS


DEFAULT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "review_text",
    "cluster_id",
    "topic_name",
)

SENTIMENT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "predicted_sentiment",
    "sentiment_label",
)

# Resolved column names used by the insights pipeline.
@dataclass(frozen=True)
class InsightsColumns:
    text: str = "review_text"
    cluster: str = "cluster_id"
    topic: str = "topic_name"
    sentiment: str = "predicted_sentiment"
    rating: str | None = "rating"

# Read-only repository for processed review analytics.
class InsightsRepository:
    def __init__(
        self,
        reviews_path: str | Path = CLUSTERED_REVIEWS_PATH,
        *,
        columns: InsightsColumns | None = None,
    ) -> None:
        self.reviews_path = Path(reviews_path)
        self.columns = columns or InsightsColumns()
        self._reviews: pd.DataFrame | None = None

    def load_reviews(self, *, refresh: bool = False) -> pd.DataFrame:
    
        if self._reviews is not None and not refresh:
            return self._reviews.copy()

        if not self.reviews_path.exists():
            raise FileNotFoundError(
                "Clustered reviews were not found at "
                f"'{self.reviews_path}'. Run the sentiment and clustering "
                "pipeline before generating insights."
            )

        reviews = pd.read_csv(self.reviews_path, low_memory=False)
        reviews = self._normalise_reviews(reviews)
        self._validate_reviews(reviews)

        self._reviews = reviews.reset_index(drop=True)
        return self._reviews.copy()

    def refresh(self) -> pd.DataFrame:
        # Reload the dataset from disk and replace the cached copy.

        return self.load_reviews(refresh=True)

    def get_topics(self) -> pd.DataFrame:
        #Return one row per topic with its review count and share.

        reviews = self.load_reviews()
        cluster_column = self.columns.cluster
        topic_column = self.columns.topic

        topics = (
            reviews.groupby([cluster_column, topic_column], dropna=False)
            .size()
            .rename("review_count")
            .reset_index()
        )

        total_reviews = int(topics["review_count"].sum())
        topics["review_share"] = (
            topics["review_count"] / total_reviews if total_reviews else 0.0
        )

        return topics.sort_values(
            ["review_count", cluster_column],
            ascending=[False, True],
        ).reset_index(drop=True)

    def get_topic_reviews(
        self,
        topic: int | str,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        # Return reviews for one topic by cluster ID or topic name.

        reviews = self.load_reviews()

        if isinstance(topic, int):
            mask = reviews[self.columns.cluster] == topic
        else:
            mask = reviews[self.columns.topic].astype(str) == str(topic)

        result = reviews.loc[mask].copy()

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1 when provided.")
            result = result.head(limit)

        return result.reset_index(drop=True)

    def get_sentiment_distribution(
        self,
        *,
        topic: int | str | None = None,
    ) -> pd.DataFrame:
        # Return sentiment counts and shares for all reviews or one topic.

        reviews = (
            self.get_topic_reviews(topic)
            if topic is not None
            else self.load_reviews()
        )
        sentiment_column = self.columns.sentiment

        counts = (
            reviews[sentiment_column]
            .value_counts(dropna=False)
            .reindex(SENTIMENT_LABELS, fill_value=0)
            .rename_axis("sentiment")
            .rename("review_count")
            .reset_index()
        )

        total_reviews = int(counts["review_count"].sum())
        counts["review_share"] = (
            counts["review_count"] / total_reviews if total_reviews else 0.0
        )

        return counts

#   Return topic-level sentiment counts and rates.
    def get_topic_sentiment(self) -> pd.DataFrame:

        reviews = self.load_reviews()
        cluster_column = self.columns.cluster
        topic_column = self.columns.topic
        sentiment_column = self.columns.sentiment

        counts = (
            reviews.groupby(
                [cluster_column, topic_column, sentiment_column],
                dropna=False,
            )
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SENTIMENT_LABELS, fill_value=0)
            .reset_index()
        )

        counts["review_count"] = counts[list(SENTIMENT_LABELS)].sum(axis=1)

        for label in SENTIMENT_LABELS:
            counts[f"{label}_rate"] = self._safe_divide(
                counts[label],
                counts["review_count"],
            )

        return counts.sort_values(
            ["negative_rate", "review_count"],
            ascending=[False, False],
        ).reset_index(drop=True)

#   Return clean review texts for use in summaries and reports.
    def get_representative_reviews(
        self,
        topic: int | str,
        *,
        sentiment: str | None = None,
        limit: int = 3,
        shortest_first: bool = False,
    ) -> list[str]:

        if limit < 1:
            raise ValueError("limit must be at least 1.")

        reviews = self.get_topic_reviews(topic)

        if sentiment is not None:
            normalised_sentiment = self._normalise_sentiment_value(sentiment)
            reviews = reviews.loc[
                reviews[self.columns.sentiment] == normalised_sentiment
            ]

        text_column = self.columns.text
        texts = (
            reviews[text_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        texts = texts[texts.ne("")].drop_duplicates()

        ordered = texts.loc[
            texts.str.len().sort_values(ascending=shortest_first).index
        ]

        return ordered.head(limit).tolist()

    # Return the main dataset-level metrics used by the dashboard.
    def get_overview(self) -> dict[str, int | float]:
        reviews = self.load_reviews()
        sentiment = self.get_sentiment_distribution().set_index("sentiment")

        overview: dict[str, int | float] = {
            "total_reviews": int(len(reviews)),
            "total_topics": int(reviews[self.columns.cluster].nunique()),
        }

        for label in SENTIMENT_LABELS:
            overview[f"{label}_reviews"] = int(
                sentiment.loc[label, "review_count"]
            )
            overview[f"{label}_rate"] = float(
                sentiment.loc[label, "review_share"]
            )

        rating_column = self.columns.rating
        if rating_column and rating_column in reviews.columns:
            numeric_ratings = pd.to_numeric(
                reviews[rating_column],
                errors="coerce",
            )
            overview["average_rating"] = float(numeric_ratings.mean())

        return overview
    
    # Raise a clear error when optional downstream columns are missing.
    def require_columns(self, columns: Iterable[str]) -> None:
        reviews = self.load_reviews()
        missing = [column for column in columns if column not in reviews.columns]

        if missing:
            raise ValueError(
                "The insights dataset is missing required columns: "
                f"{missing}. Available columns: {list(reviews.columns)}"
            )

    def _normalise_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        output = reviews.copy()

        sentiment_column = self._resolve_sentiment_column(output.columns)
        if sentiment_column != self.columns.sentiment:
            output = output.rename(
                columns={sentiment_column: self.columns.sentiment}
            )

        output[self.columns.text] = (
            output[self.columns.text]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        output[self.columns.cluster] = pd.to_numeric(
            output[self.columns.cluster],
            errors="raise",
        ).astype(int)

        output[self.columns.topic] = (
            output[self.columns.topic]
            .fillna("General / uncategorized")
            .astype(str)
            .str.strip()
            .replace("", "General / uncategorized")
        )

        output[self.columns.sentiment] = output[
            self.columns.sentiment
        ].map(self._normalise_sentiment_value)

        return output

    def _validate_reviews(self, reviews: pd.DataFrame) -> None:
        required = list(DEFAULT_REQUIRED_COLUMNS) + [self.columns.sentiment]
        missing = [column for column in required if column not in reviews.columns]

        if missing:
            raise ValueError(
                "The clustered review dataset is missing required columns: "
                f"{missing}. Available columns: {list(reviews.columns)}"
            )

        if reviews.empty:
            raise ValueError("The clustered review dataset is empty.")

        if reviews[self.columns.text].eq("").all():
            raise ValueError("The review_text column contains no usable text.")

        invalid_sentiments = sorted(
            set(reviews[self.columns.sentiment].dropna().unique())
            - set(SENTIMENT_LABELS)
        )
        if invalid_sentiments:
            raise ValueError(
                "Unsupported sentiment values found: "
                f"{invalid_sentiments}. Expected one of {SENTIMENT_LABELS}."
            )

    def _resolve_sentiment_column(self, columns: Sequence[str]) -> str:
        if self.columns.sentiment in columns:
            return self.columns.sentiment

        for candidate in SENTIMENT_COLUMN_CANDIDATES:
            if candidate in columns:
                return candidate

        raise ValueError(
            "No sentiment column was found. Expected one of "
            f"{list(SENTIMENT_COLUMN_CANDIDATES)}."
        )

    @staticmethod
    def _normalise_sentiment_value(value: object) -> str:
        label = str(value).strip().lower()

        aliases = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "0": "negative",
            "1": "neutral",
            "2": "positive",
        }

        return aliases.get(label, label)

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        return numerator.div(denominator.where(denominator.ne(0), 1)).fillna(0.0)
