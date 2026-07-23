"""
This module combines:
- repository
- summaries
- recommendations

into a single report that can be exported to JSON,
Markdown or served through FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from .repository import InsightsRepository
from .summary import (
    generate_overview_summary,
    generate_topic_summaries,
)
from .recommendation import (
    generate_all_recommendations,
)


# ==========================================================
# Executive Summary
# ==========================================================

def generate_executive_summary(
    repository: InsightsRepository,
) -> str:
    """
    Create a short executive summary for business users.
    """

    overview = generate_overview_summary(repository)

    return (
        f"VoxForge AI analysed "
        f"{overview['total_reviews']:,} customer reviews "
        f"covering {overview['total_topics']} major product topics. "
        f"The overall customer sentiment is "
        f"{overview['overall_sentiment'].lower()}, "
        f"with {overview['positive_percentage']:.1f}% positive reviews "
        f"and {overview['negative_percentage']:.1f}% negative reviews. "
        f"The dashboard highlights the most important product areas "
        f"requiring business attention."
    )


# ==========================================================
# Key Findings
# ==========================================================

def generate_key_findings(
    repository: InsightsRepository,
) -> List[str]:
    """
    Generate high-level findings from the analysis.
    """

    summaries = generate_topic_summaries(repository)

    findings = []

    for topic in summaries:

        findings.append(
            f"{topic['topic']} "
            f"contains {topic['topic_size']} reviews "
            f"with {topic['positive_ratio']:.0%} positive sentiment."
        )

    return findings


# ==========================================================
# Business Risks
# ==========================================================

def generate_business_risks(
    repository: InsightsRepository,
) -> List[str]:
    """
    Extract the most critical business risks.
    """

    recommendations = generate_all_recommendations(repository)

    risks = []

    for item in recommendations:

        if item["priority"] != "High":
            continue

        risks.append(
            f"{item['topic']}: "
            f"{item['recommendation']}"
        )

    return risks


# ==========================================================
# Business Opportunities
# ==========================================================

def generate_business_opportunities(
    repository: InsightsRepository,
) -> List[str]:
    """
    Extract positive opportunities from the analysis.
    """

    summaries = generate_topic_summaries(repository)

    opportunities = []

    for topic in summaries:

        if topic["positive_ratio"] < 0.70:
            continue

        opportunities.append(
            f"Customers consistently report positive "
            f"experiences with {topic['topic']}. "
            f"This product area can be highlighted in "
            f"marketing and customer success initiatives."
        )

    return opportunities


# ==========================================================
# Overall Report Metrics
# ==========================================================

def generate_report_metrics(
    repository: InsightsRepository,
) -> Dict:

    overview = generate_overview_summary(repository)

    recommendations = generate_all_recommendations(repository)

    return {

        "generated_at": datetime.utcnow().isoformat(),

        "dataset": {

            "reviews": overview["total_reviews"],

            "topics": overview["total_topics"]

        },

        "sentiment": {

            "positive": overview["positive_percentage"],

            "negative": overview["negative_percentage"],

            "neutral": overview["neutral_percentage"]

        },

        "recommendations": len(recommendations),

        "high_priority":

            len(

                [

                    r

                    for r in recommendations

                    if r["priority"] == "High"

                ]

            )

    }


# ==========================================================
# Complete report payload
# ==========================================================

def generate_report(
    repository: InsightsRepository,
) -> Dict:
    """
    Generate the complete executive report payload.

    The returned dictionary is JSON serializable and can be:
    - served through FastAPI
    - written to a JSON artifact
    - rendered as Markdown
    """

    overview = generate_overview_summary(repository)
    topics = generate_topic_summaries(repository)
    recommendations = generate_all_recommendations(repository)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "executive_summary": generate_executive_summary(repository),
        "overview": overview,
        "metrics": generate_report_metrics(repository),
        "key_findings": generate_key_findings(repository),
        "business_risks": generate_business_risks(repository),
        "business_opportunities": generate_business_opportunities(repository),
        "topics": topics,
        "recommendations": recommendations,
    }


# ==========================================================
# Markdown helpers
# ==========================================================

def _format_percentage(value: float) -> str:
    """
    Format either a ratio or an already-calculated percentage.

    Examples:
    0.72 -> 72.0%
    72.0 -> 72.0%
    """

    numeric_value = float(value)

    if numeric_value <= 1:
        numeric_value *= 100

    return f"{numeric_value:.1f}%"


def _format_list_section(
    title: str,
    items: List[str],
    empty_message: str,
) -> List[str]:
    """
    Render a Markdown section containing bullet points.
    """

    lines = [
        f"## {title}",
        "",
    ]

    if not items:
        lines.append(empty_message)
        lines.append("")
        return lines

    for item in items:
        lines.append(f"- {item}")

    lines.append("")

    return lines


# ==========================================================
# Markdown report
# ==========================================================

def generate_markdown_report(
    repository: InsightsRepository,
) -> str:
    """
    Render the complete executive report as Markdown.
    """

    report = generate_report(repository)

    overview = report["overview"]
    metrics = report["metrics"]
    sentiment = metrics["sentiment"]

    lines: List[str] = [
        "# VoxForge AI Executive Review Report",
        "",
        f"_Generated at: {report['generated_at']}_",
        "",
        "## Executive Summary",
        "",
        report["executive_summary"],
        "",
        "## Dataset Overview",
        "",
        f"- Total reviews: {overview['total_reviews']:,}",
        f"- Total topics: {overview['total_topics']}",
        f"- Overall sentiment: {overview['overall_sentiment']}",
        f"- Positive sentiment: "
        f"{_format_percentage(sentiment['positive'])}",
        f"- Negative sentiment: "
        f"{_format_percentage(sentiment['negative'])}",
        f"- Neutral sentiment: "
        f"{_format_percentage(sentiment['neutral'])}",
        f"- Total recommendations: "
        f"{metrics['recommendations']}",
        f"- High-priority recommendations: "
        f"{metrics['high_priority']}",
        "",
    ]

    lines.extend(
        _format_list_section(
            title="Key Findings",
            items=report["key_findings"],
            empty_message="No key findings were generated.",
        )
    )

    lines.extend(
        _format_list_section(
            title="Business Risks",
            items=report["business_risks"],
            empty_message=(
                "No high-priority business risks were detected."
            ),
        )
    )

    lines.extend(
        _format_list_section(
            title="Business Opportunities",
            items=report["business_opportunities"],
            empty_message=(
                "No strong positive opportunities were detected."
            ),
        )
    )

    lines.extend(
        [
            "## Topic Analysis",
            "",
        ]
    )

    topics = report["topics"]

    if not topics:
        lines.extend(
            [
                "No topic summaries were generated.",
                "",
            ]
        )
    else:
        for topic in topics:
            topic_name = topic.get(
                "topic",
                f"Topic {topic.get('topic_id', '')}",
            )

            lines.extend(
                [
                    f"### {topic_name}",
                    "",
                    f"- Topic ID: {topic.get('topic_id')}",
                    f"- Reviews: {topic.get('topic_size', 0):,}",
                    f"- Positive sentiment: "
                    f"{_format_percentage(topic.get('positive_ratio', 0))}",
                    f"- Negative sentiment: "
                    f"{_format_percentage(topic.get('negative_ratio', 0))}",
                    f"- Sentiment health: "
                    f"{topic.get('sentiment_health', 'Unknown')}",
                    "",
                ]
            )

            narrative = topic.get(
                "summary",
                topic.get("narrative"),
            )

            if narrative:
                lines.extend(
                    [
                        narrative,
                        "",
                    ]
                )

            positive_examples = topic.get(
                "representative_positive_reviews",
                [],
            )

            negative_examples = topic.get(
                "representative_negative_reviews",
                [],
            )

            if positive_examples:
                lines.extend(
                    [
                        "**Representative positive reviews**",
                        "",
                    ]
                )

                for review in positive_examples:
                    if isinstance(review, dict):
                        review_text = review.get(
                            "text",
                            review.get(
                                "review",
                                str(review),
                            ),
                        )
                    else:
                        review_text = str(review)

                    lines.append(f"- {review_text}")

                lines.append("")

            if negative_examples:
                lines.extend(
                    [
                        "**Representative negative reviews**",
                        "",
                    ]
                )

                for review in negative_examples:
                    if isinstance(review, dict):
                        review_text = review.get(
                            "text",
                            review.get(
                                "review",
                                str(review),
                            ),
                        )
                    else:
                        review_text = str(review)

                    lines.append(f"- {review_text}")

                lines.append("")

    lines.extend(
        [
            "## Recommendations",
            "",
        ]
    )

    recommendations = report["recommendations"]

    if not recommendations:
        lines.extend(
            [
                "No recommendations were generated.",
                "",
            ]
        )
    else:
        for item in recommendations:
            lines.extend(
                [
                    f"### {item['rank']}. {item['topic']}",
                    "",
                    f"- Priority: {item['priority']}",
                    f"- Severity score: {item['severity']}",
                    f"- Topic size: {item['topic_size']:,}",
                    f"- Positive sentiment: "
                    f"{_format_percentage(item['positive_ratio'])}",
                    f"- Negative sentiment: "
                    f"{_format_percentage(item['negative_ratio'])}",
                    "",
                    f"**Recommendation:** "
                    f"{item['recommendation']}",
                    "",
                    f"**Rationale:** {item['rationale']}",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


# ==========================================================
# JSON-ready public helper
# ==========================================================

#   Return the report as a JSON-ready dictionary.
def generate_json_report(
    repository: InsightsRepository,
) -> Dict:

    return generate_report(repository)


# ==========================================================
# Public convenience function
# ==========================================================

def build_executive_report(
    repository: InsightsRepository | None = None,
) -> Dict:

    active_repository = (
        repository
        if repository is not None
        else InsightsRepository()
    )

    return generate_report(active_repository)
