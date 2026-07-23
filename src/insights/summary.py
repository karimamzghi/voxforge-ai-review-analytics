"""
This is a feterministic business summaries module that converts repository aggregates into plain Python dictionaries that
can be exported to JSON and served by FastAPI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.insights.repository import InsightsRepository


HIGH_RATE = 0.60
MODERATE_RATE = 0.35
LOW_RATE = 0.15

# Convert a decimal rate to a rounded percentage.
def _percentage(value: float) -> float:
    return round(float(value) * 100, 1)


# Return the sentiment label with the highest rate for one topic.
def _dominant_sentiment(row: pd.Series) -> str:
    rates = {
        "positive": float(row.get("positive_rate", 0.0)),
        "neutral": float(row.get("neutral_rate", 0.0)),
        "negative": float(row.get("negative_rate", 0.0)),
    }
    return max(rates, key=rates.get)


# Return a simple business-facing health classification.
def _sentiment_health(
    positive_rate: float,
    negative_rate: float,
) -> str:

    if negative_rate >= HIGH_RATE:
        return "critical"
    if negative_rate >= MODERATE_RATE:
        return "needs_attention"
    if positive_rate >= HIGH_RATE and negative_rate < LOW_RATE:
        return "strong"
    return "mixed"


# Create a concise deterministic narrative for one topic.
def _topic_narrative(
    topic_name: str,
    review_count: int,
    review_share: float,
    positive_rate: float,
    neutral_rate: float,
    negative_rate: float,
) -> str:
    
    share_pct = _percentage(review_share)
    positive_pct = _percentage(positive_rate)
    neutral_pct = _percentage(neutral_rate)
    negative_pct = _percentage(negative_rate)

    if negative_rate >= HIGH_RATE:
        insight = (
            f"Customer feedback is predominantly negative ({negative_pct}%), "
            "making this a high-priority improvement area."
        )
    elif negative_rate >= MODERATE_RATE:
        insight = (
            f"Negative feedback is material ({negative_pct}%) and should be "
            "reviewed alongside the positive experiences."
        )
    elif positive_rate >= HIGH_RATE:
        insight = (
            f"Customer feedback is strongly positive ({positive_pct}%), "
            "indicating a clear product strength."
        )
    else:
        insight = (
            f"Feedback is mixed: {positive_pct}% positive, {neutral_pct}% "
            f"neutral, and {negative_pct}% negative."
        )

    return (
        f"{topic_name} contains {review_count:,} reviews, representing "
        f"{share_pct}% of the analysed dataset. {insight}"
    )


# Generate a dashboard-ready summary for one topic.
def generate_topic_summary(
    repository: InsightsRepository,
    topic: int | str,
    *,
    representative_review_limit: int = 3,
) -> dict[str, Any]:

    if representative_review_limit < 1:
        raise ValueError("representative_review_limit must be at least 1.")

    topic_metrics = repository.get_topic_sentiment()
    cluster_column = repository.columns.cluster
    topic_column = repository.columns.topic

    if isinstance(topic, int):
        matches = topic_metrics.loc[topic_metrics[cluster_column] == topic]
    else:
        matches = topic_metrics.loc[
            topic_metrics[topic_column].astype(str) == str(topic)
        ]

    if matches.empty:
        raise ValueError(f"Topic '{topic}' was not found in the insights dataset.")

    row = matches.iloc[0]
    overview = repository.get_overview()

    review_count = int(row["review_count"])
    total_reviews = int(overview["total_reviews"])
    review_share = review_count / total_reviews if total_reviews else 0.0

    positive_rate = float(row.get("positive_rate", 0.0))
    neutral_rate = float(row.get("neutral_rate", 0.0))
    negative_rate = float(row.get("negative_rate", 0.0))

    topic_id = int(row[cluster_column])
    topic_name = str(row[topic_column])
    dominant_sentiment = _dominant_sentiment(row)

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "review_count": review_count,
        "review_share": round(review_share, 4),
        "review_share_percent": _percentage(review_share),
        "sentiment": {
            "dominant": dominant_sentiment,
            "health": _sentiment_health(positive_rate, negative_rate),
            "positive_count": int(row.get("positive", 0)),
            "neutral_count": int(row.get("neutral", 0)),
            "negative_count": int(row.get("negative", 0)),
            "positive_rate": round(positive_rate, 4),
            "neutral_rate": round(neutral_rate, 4),
            "negative_rate": round(negative_rate, 4),
            "positive_percent": _percentage(positive_rate),
            "neutral_percent": _percentage(neutral_rate),
            "negative_percent": _percentage(negative_rate),
        },
        "summary": _topic_narrative(
            topic_name=topic_name,
            review_count=review_count,
            review_share=review_share,
            positive_rate=positive_rate,
            neutral_rate=neutral_rate,
            negative_rate=negative_rate,
        ),
        "representative_reviews": {
            "positive": repository.get_representative_reviews(
                topic_id,
                sentiment="positive",
                limit=representative_review_limit,
            ),
            "negative": repository.get_representative_reviews(
                topic_id,
                sentiment="negative",
                limit=representative_review_limit,
            ),
        },
    }


# Generate summaries for every topic ordered by review volume.
def generate_topic_summaries(
    repository: InsightsRepository,
    *,
    representative_review_limit: int = 3,
) -> list[dict[str, Any]]:

    topics = repository.get_topics()
    cluster_column = repository.columns.cluster

    summaries = [
        generate_topic_summary(
            repository,
            int(row[cluster_column]),
            representative_review_limit=representative_review_limit,
        )
        for _, row in topics.iterrows()
    ]

    return summaries


# Generate the dataset-level summary used by the dashboard hero and KPIs.
def generate_overview_summary(
    repository: InsightsRepository,
) -> dict[str, Any]:

    overview = repository.get_overview()
    topic_summaries = generate_topic_summaries(
        repository,
        representative_review_limit=1,
    )

    if not topic_summaries:
        raise ValueError("No topic summaries could be generated.")

    largest_topic = max(topic_summaries, key=lambda item: item["review_count"])
    most_negative = max(
        topic_summaries,
        key=lambda item: item["sentiment"]["negative_rate"],
    )
    most_positive = max(
        topic_summaries,
        key=lambda item: item["sentiment"]["positive_rate"],
    )

    positive_rate = float(overview.get("positive_rate", 0.0))
    neutral_rate = float(overview.get("neutral_rate", 0.0))
    negative_rate = float(overview.get("negative_rate", 0.0))

    narrative = (
        f"VoxForge analysed {int(overview['total_reviews']):,} reviews across "
        f"{int(overview['total_topics'])} topics. Overall sentiment is "
        f"{_percentage(positive_rate)}% positive, "
        f"{_percentage(neutral_rate)}% neutral, and "
        f"{_percentage(negative_rate)}% negative. "
        f"{largest_topic['topic_name']} is the largest topic, while "
        f"{most_negative['topic_name']} has the highest negative sentiment rate."
    )

    output: dict[str, Any] = {
        **overview,
        "positive_percent": _percentage(positive_rate),
        "neutral_percent": _percentage(neutral_rate),
        "negative_percent": _percentage(negative_rate),
        "summary": narrative,
        "largest_topic": {
            "topic_id": largest_topic["topic_id"],
            "topic_name": largest_topic["topic_name"],
            "review_count": largest_topic["review_count"],
            "review_share_percent": largest_topic["review_share_percent"],
        },
        "most_negative_topic": {
            "topic_id": most_negative["topic_id"],
            "topic_name": most_negative["topic_name"],
            "negative_percent": most_negative["sentiment"]["negative_percent"],
        },
        "most_positive_topic": {
            "topic_id": most_positive["topic_id"],
            "topic_name": most_positive["topic_name"],
            "positive_percent": most_positive["sentiment"]["positive_percent"],
        },
    }

    if "average_rating" in overview:
        average_rating = float(overview["average_rating"])
        output["average_rating"] = (
            round(average_rating, 2) if pd.notna(average_rating) else None
        )

    return output


__all__ = [
    "generate_overview_summary",
    "generate_topic_summary",
    "generate_topic_summaries",
]
