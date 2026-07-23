from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class Metric(BaseModel):
    label: str
    value: str
    detail: str


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float


class Topic(BaseModel):
    id: int
    name: str
    description: str
    review_count: int
    sentiment: SentimentBreakdown
    keywords: list[str]
    representative_reviews: list[str]
    recommendation: str


class Recommendation(BaseModel):
    id: int
    title: str
    priority: str
    topic: str
    evidence: str
    action: str


class DashboardData(BaseModel):
    metrics: list[Metric]
    overall_sentiment: SentimentBreakdown
    topics: list[Topic]
    recommendations: list[Recommendation]
    executive_summary: str
