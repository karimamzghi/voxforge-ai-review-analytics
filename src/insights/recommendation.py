"""
recommendation.py

Generate business recommendations from customer review topics.

Author: VoxForge AI
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict

import pandas as pd

from .repository import InsightsRepository


# ==========================================================
# Recommendation model
# ==========================================================

@dataclass
class Recommendation:
    topic: str
    priority: str
    severity: float
    recommendation: str
    rationale: str
    positive_ratio: float
    negative_ratio: float

    def to_dict(self):
        return asdict(self)


# ==========================================================
# Thresholds
# ==========================================================

HIGH_NEGATIVE = 0.50
MEDIUM_NEGATIVE = 0.35
LOW_NEGATIVE = 0.20

HIGH_POSITIVE = 0.70


# ==========================================================
# Priority calculation
# ==========================================================

def calculate_priority(negative_ratio: float) -> str:
    """
    Convert a negative review ratio into a business priority.
    """

    if negative_ratio >= HIGH_NEGATIVE:
        return "High"

    if negative_ratio >= MEDIUM_NEGATIVE:
        return "Medium"

    return "Low"


# ==========================================================
# Severity score
# ==========================================================

def calculate_severity(
    topic_size: int,
    negative_ratio: float,
) -> float:
    """
    Score between 0 and 100.

    Larger topics containing many negative reviews
    should appear first in the dashboard.
    """

    score = (
        topic_size * 0.40
        + negative_ratio * 100 * 0.60
    )

    return round(score, 1)


# ==========================================================
# Recommendation templates
# ==========================================================

NEGATIVE_ACTIONS = {

    "Battery": (
        "Investigate battery life complaints and review hardware quality.",
        "Battery-related issues frequently lead to poor customer satisfaction."
    ),

    "Echo": (
        "Review Alexa responsiveness, connectivity, and firmware stability.",
        "Voice assistant reliability has a direct impact on daily usage."
    ),

    "Kindle": (
        "Improve reading experience, software stability, and synchronization.",
        "Reading interruptions reduce perceived product quality."
    ),

    "Fire TV": (
        "Optimize streaming performance and application responsiveness.",
        "Entertainment devices are highly sensitive to lag and crashes."
    ),

    "Fire Tablet": (
        "Focus on device speed, durability, and battery optimisation.",
        "Tablet users report performance issues affecting everyday usage."
    ),

    "General": (
        "Perform deeper qualitative analysis to identify recurring pain points.",
        "This cluster combines reviews without a dominant theme."
    ),
}


POSITIVE_ACTION = (
    "Maintain the current customer experience while continuing to monitor future reviews.",
    "Customers generally report positive experiences in this topic."
)


# ==========================================================
# Template selection
# ==========================================================

def recommendation_template(
    topic_name: str,
    negative_ratio: float,
):
    """
    Choose the correct recommendation text.
    """

    if negative_ratio < LOW_NEGATIVE:
        return POSITIVE_ACTION

    for keyword, value in NEGATIVE_ACTIONS.items():

        if keyword.lower() in topic_name.lower():
            return value

    return (
        "Investigate recurring customer concerns and prioritize product improvements.",
        "Customer feedback indicates opportunities for quality improvements."
    )


# ==========================================================
# Formatting
# ==========================================================

def build_recommendation(
    topic: str,
    topic_size: int,
    positive_ratio: float,
    negative_ratio: float,
) -> Recommendation:

    recommendation, rationale = recommendation_template(
        topic,
        negative_ratio,
    )

    severity = calculate_severity(
        topic_size,
        negative_ratio,
    )

    priority = calculate_priority(
        negative_ratio,
    )

    return Recommendation(
        topic=topic,
        priority=priority,
        severity=severity,
        recommendation=recommendation,
        rationale=rationale,
        positive_ratio=round(positive_ratio, 2),
        negative_ratio=round(negative_ratio, 2),
    )
# ==========================================================
# Data preparation
# ==========================================================

def _find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return the first matching column name from a list of candidates.
    """

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def _prepare_topic_metrics(
    repository: InsightsRepository,
) -> pd.DataFrame:
    """
    Build one normalized dataframe containing topic-level metrics.

    The helper accepts slightly different column names so the
    recommendation layer remains compatible with notebook exports.
    """

    topics = repository.get_topics()
    sentiment = repository.get_topic_sentiment()

    if topics.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic",
                "topic_size",
                "positive_ratio",
                "negative_ratio",
            ]
        )

    topic_id_column = _find_column(
        topics,
        ["topic_id", "cluster", "cluster_id"],
    )
    topic_name_column = _find_column(
        topics,
        ["topic", "topic_name", "cluster_name", "label"],
    )
    topic_size_column = _find_column(
        topics,
        ["topic_size", "review_count", "count", "size"],
    )

    if topic_id_column is None:
        raise ValueError(
            "Topic data must contain one of: "
            "'topic_id', 'cluster', or 'cluster_id'."
        )

    if topic_name_column is None:
        topics = topics.copy()
        topics["topic"] = topics[topic_id_column].apply(
            lambda topic_id: f"Topic {topic_id}"
        )
        topic_name_column = "topic"

    if topic_size_column is None:
        topics = topics.copy()
        topics["topic_size"] = 0
        topic_size_column = "topic_size"

    positive_column = _find_column(
        sentiment,
        [
            "positive_ratio",
            "positive_percentage",
            "positive_share",
            "positive",
        ],
    )
    negative_column = _find_column(
        sentiment,
        [
            "negative_ratio",
            "negative_percentage",
            "negative_share",
            "negative",
        ],
    )
    sentiment_topic_id_column = _find_column(
        sentiment,
        ["topic_id", "cluster", "cluster_id"],
    )

    if sentiment_topic_id_column is None:
        raise ValueError(
            "Topic sentiment data must contain one of: "
            "'topic_id', 'cluster', or 'cluster_id'."
        )

    sentiment = sentiment.copy()

    if positive_column is None:
        sentiment["positive_ratio"] = 0.0
        positive_column = "positive_ratio"

    if negative_column is None:
        sentiment["negative_ratio"] = 0.0
        negative_column = "negative_ratio"

    metrics = topics[
        [
            topic_id_column,
            topic_name_column,
            topic_size_column,
        ]
    ].copy()

    metrics.columns = [
        "topic_id",
        "topic",
        "topic_size",
    ]

    sentiment_metrics = sentiment[
        [
            sentiment_topic_id_column,
            positive_column,
            negative_column,
        ]
    ].copy()

    sentiment_metrics.columns = [
        "topic_id",
        "positive_ratio",
        "negative_ratio",
    ]

    metrics = metrics.merge(
        sentiment_metrics,
        on="topic_id",
        how="left",
    )

    metrics["topic_size"] = (
        pd.to_numeric(metrics["topic_size"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for column in ["positive_ratio", "negative_ratio"]:
        metrics[column] = pd.to_numeric(
            metrics[column],
            errors="coerce",
        ).fillna(0.0)

        # Convert percentages such as 72.5 into ratios such as 0.725.
        if metrics[column].max() > 1:
            metrics[column] = metrics[column] / 100

        metrics[column] = metrics[column].clip(
            lower=0,
            upper=1,
        )

    return metrics


# ==========================================================
# Recommendation generation
# ==========================================================

def generate_topic_recommendation(
    repository: InsightsRepository,
    topic_id: int,
) -> dict:
    """
    Generate one recommendation for a selected topic.
    """

    metrics = _prepare_topic_metrics(repository)

    topic_row = metrics.loc[
        metrics["topic_id"] == topic_id
    ]

    if topic_row.empty:
        raise ValueError(
            f"No topic was found for topic_id={topic_id}."
        )

    row = topic_row.iloc[0]

    recommendation = build_recommendation(
        topic=str(row["topic"]),
        topic_size=int(row["topic_size"]),
        positive_ratio=float(row["positive_ratio"]),
        negative_ratio=float(row["negative_ratio"]),
    )

    result = recommendation.to_dict()
    result["topic_id"] = int(row["topic_id"])
    result["topic_size"] = int(row["topic_size"])

    return result

# Generate recommendations for all discovered topics. 
# Recommendations are sorted by severity so the most important
# business issues appear first in the dashboard.
def generate_all_recommendations(
    repository: InsightsRepository,
) -> list[dict]:

    metrics = _prepare_topic_metrics(repository)

    recommendations: list[dict] = []

    for row in metrics.itertuples(index=False):
        recommendation = build_recommendation(
            topic=str(row.topic),
            topic_size=int(row.topic_size),
            positive_ratio=float(row.positive_ratio),
            negative_ratio=float(row.negative_ratio),
        )

        item = recommendation.to_dict()
        item["topic_id"] = int(row.topic_id)
        item["topic_size"] = int(row.topic_size)

        recommendations.append(item)

    recommendations.sort(
        key=lambda item: (
            item["severity"],
            item["negative_ratio"],
            item["topic_size"],
        ),
        reverse=True,
    )

    for index, item in enumerate(
        recommendations,
        start=1,
    ):
        item["rank"] = index

    return recommendations


# ==========================================================
# Dashboard helpers
# ==========================================================

def get_priority_counts(
    recommendations: list[dict],
) -> dict[str, int]:
    """
    Count recommendations by business priority.
    """

    counts = {
        "High": 0,
        "Medium": 0,
        "Low": 0,
    }

    for recommendation in recommendations:
        priority = recommendation.get(
            "priority",
            "Low",
        )

        if priority not in counts:
            counts[priority] = 0

        counts[priority] += 1

    return counts


def get_top_recommendations(
    recommendations: list[dict],
    limit: int = 3,
) -> list[dict]:
    """
    Return the highest-severity recommendations.
    """

    if limit < 1:
        return []

    return recommendations[:limit]


def generate_recommendation_summary(
    recommendations: list[dict],
) -> dict:
    """
    Create summary metrics for recommendation dashboard cards.
    """

    if not recommendations:
        return {
            "total_recommendations": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0,
            "average_severity": 0.0,
            "highest_risk_topic": None,
        }

    priority_counts = get_priority_counts(
        recommendations
    )

    average_severity = sum(
        item["severity"]
        for item in recommendations
    ) / len(recommendations)

    highest_risk = max(
        recommendations,
        key=lambda item: item["severity"],
    )

    return {
        "total_recommendations": len(
            recommendations
        ),
        "high_priority": priority_counts.get(
            "High",
            0,
        ),
        "medium_priority": priority_counts.get(
            "Medium",
            0,
        ),
        "low_priority": priority_counts.get(
            "Low",
            0,
        ),
        "average_severity": round(
            average_severity,
            1,
        ),
        "highest_risk_topic": {
            "topic_id": highest_risk["topic_id"],
            "topic": highest_risk["topic"],
            "severity": highest_risk["severity"],
            "priority": highest_risk["priority"],
        },
    }


def generate_recommendation_payload(
    repository: InsightsRepository,
    top_limit: int = 3,
) -> dict:
    """
    Generate the complete JSON-ready recommendation payload.
    """

    recommendations = generate_all_recommendations(
        repository
    )

    return {
        "summary": generate_recommendation_summary(
            recommendations
        ),
        "top_recommendations": get_top_recommendations(
            recommendations,
            limit=top_limit,
        ),
        "recommendations": recommendations,
    }

