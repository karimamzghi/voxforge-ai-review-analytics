"""
These schemas:
- document the API contract
- improve generated FastAPI documentation
- validate response payloads
- keep routes and frontend integration predictable

The models are intentionally flexible because exported analytics
artifacts may contain additional fields as the project evolves.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


# ==========================================================
# Shared base model
# ==========================================================

class APIModel(BaseModel):
    """
    Base model used by all API schemas.

    extra="allow" keeps the API compatible with additional analytics
    fields added to exported artifacts later.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


# ==========================================================
# Generic API responses
# ==========================================================

class HealthResponse(APIModel):
    status: str
    service: str


class RootResponse(APIModel):
    application: str
    status: str
    version: str
    documentation: str


class ErrorResponse(APIModel):
    detail: str


# ==========================================================
# Sentiment schemas
# ==========================================================

class SentimentDistribution(APIModel):
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0


class SentimentCounts(APIModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0


# ==========================================================
# Dashboard overview
# ==========================================================

class DashboardOverview(APIModel):
    total_reviews: int = 0
    total_topics: int = 0
    overall_sentiment: str = "Unknown"

    positive_percentage: float = 0.0
    negative_percentage: float = 0.0
    neutral_percentage: float = 0.0

    positive_reviews: int | None = None
    negative_reviews: int | None = None
    neutral_reviews: int | None = None


# ==========================================================
# Representative review schemas
# ==========================================================

class RepresentativeReview(APIModel):
    """
    One representative customer review.

    Depending on the analytics pipeline, a review may contain only text
    or additional fields such as sentiment, score, review ID or date.
    """

    text: str | None = None
    review: str | None = None
    sentiment: str | None = None
    score: float | None = None
    review_id: str | int | None = None


# ==========================================================
# Topic schemas
# ==========================================================

class TopicSummary(APIModel):
    topic_id: int | str
    topic: str

    topic_size: int = 0

    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0

    sentiment_health: str = "Unknown"

    summary: str | None = None
    narrative: str | None = None

    representative_positive_reviews: list[
        RepresentativeReview | str
    ] = Field(default_factory=list)

    representative_negative_reviews: list[
        RepresentativeReview | str
    ] = Field(default_factory=list)

    representative_neutral_reviews: list[
        RepresentativeReview | str
    ] = Field(default_factory=list)


class TopicListResponse(APIModel):
    topics: list[TopicSummary] = Field(default_factory=list)
    count: int = 0


# ==========================================================
# Recommendation schemas
# ==========================================================

class Recommendation(APIModel):
    topic_id: int | str
    topic: str

    rank: int | None = None
    priority: str = "Low"
    severity: float = 0.0

    topic_size: int = 0

    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0

    recommendation: str
    rationale: str | None = None


class RecommendationPriorityCounts(APIModel):
    high: int = 0
    medium: int = 0
    low: int = 0


class RecommendationSummary(APIModel):
    total_recommendations: int = 0
    high_priority: int = 0
    medium_priority: int = 0
    low_priority: int = 0


class RecommendationPayload(APIModel):
    recommendations: list[Recommendation] = Field(
        default_factory=list
    )

    priority_counts: (
        RecommendationPriorityCounts
        | dict[str, int]
        | None
    ) = None

    summary: RecommendationSummary | dict[str, Any] | None = None


# ==========================================================
# Dashboard payload
# ==========================================================

class DashboardResponse(APIModel):
    overview: DashboardOverview
    topics: list[TopicSummary] = Field(default_factory=list)

    recommendations: (
        RecommendationPayload
        | list[Recommendation]
        | dict[str, Any]
    )


# ==========================================================
# Report schemas
# ==========================================================

class ReportDatasetMetrics(APIModel):
    reviews: int = 0
    topics: int = 0


class ReportMetrics(APIModel):
    generated_at: str

    dataset: ReportDatasetMetrics

    sentiment: SentimentDistribution

    recommendations: int = 0
    high_priority: int = 0


class ExecutiveReport(APIModel):
    generated_at: str

    executive_summary: str

    overview: DashboardOverview

    metrics: ReportMetrics

    key_findings: list[str] = Field(default_factory=list)
    business_risks: list[str] = Field(default_factory=list)
    business_opportunities: list[str] = Field(
        default_factory=list
    )

    topics: list[TopicSummary] = Field(default_factory=list)

    recommendations: list[Recommendation] = Field(
        default_factory=list
    )


class MarkdownReportResponse(APIModel):
    markdown: str


# ==========================================================
# Artifact schemas
# ==========================================================

class ArtifactMetadata(APIModel):
    path: str
    exists: bool
    size_bytes: int | None = None
    modified_at: float | None = None


class ArtifactStatusResponse(APIModel):
    dashboard: ArtifactMetadata
    topics: ArtifactMetadata
    recommendations: ArtifactMetadata
    report_json: ArtifactMetadata
    report_markdown: ArtifactMetadata


class ArtifactRefreshResponse(APIModel):
    dashboard: str
    topics: str
    recommendations: str
    report_json: str
    report_markdown: str


# ==========================================================
# Search and filtering schemas
# ==========================================================

class TopicSearchResponse(APIModel):
    query: str
    count: int
    results: list[TopicSummary] = Field(default_factory=list)


class RecommendationFilterResponse(APIModel):
    priority: str
    count: int
    results: list[Recommendation] = Field(default_factory=list)
