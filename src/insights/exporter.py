"""
Export insights into JSON and Markdown artifacts.

These artifacts are later consumed by:
- FastAPI
- React dashboard
- Executive reports
"""

from __future__ import annotations

import json
from pathlib import Path

from .repository import InsightsRepository
from .summary import (
    generate_overview_summary,
    generate_topic_summaries,
)
from .recommendation import (
    generate_recommendation_payload,
)
from .report import (
    generate_report,
    generate_markdown_report,
)


# ==========================================================
# JSON helpers
# ==========================================================

def save_json(data: dict, output_path: Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_markdown(
    markdown: str,
    output_path: Path,
) -> None:
    """
    Save Markdown report.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )


# ==========================================================
# Dashboard Export
# ==========================================================

def export_dashboard(
    repository: InsightsRepository,
    output_directory: Path,
) -> Path:
    """
    Export dashboard.json
    """

    dashboard = {

        "overview": generate_overview_summary(
            repository
        ),

        "topics": generate_topic_summaries(
            repository
        ),

        "recommendations":

            generate_recommendation_payload(
                repository
            )

    }

    output_file = (
        output_directory /
        "dashboard.json"
    )

    save_json(
        dashboard,
        output_file,
    )

    return output_file


# ==========================================================
# Topic Export
# ==========================================================

def export_topics(
    repository: InsightsRepository,
    output_directory: Path,
) -> Path:

    topics = generate_topic_summaries(
        repository
    )

    output_file = (
        output_directory /
        "topics.json"
    )

    save_json(
        topics,
        output_file,
    )

    return output_file


# ==========================================================
# Recommendation Export
# ==========================================================

def export_recommendations(
    repository: InsightsRepository,
    output_directory: Path,
) -> Path:

    recommendations = (
        generate_recommendation_payload(
            repository
        )
    )

    output_file = (
        output_directory /
        "recommendations.json"
    )

    save_json(
        recommendations,
        output_file,
    )

    return output_file


# ==========================================================
# Report Export
# ==========================================================

def export_report(
    repository: InsightsRepository,
    output_directory: Path,
) -> tuple[Path, Path]:

    report = generate_report(
        repository
    )

    markdown = generate_markdown_report(
        repository
    )

    json_file = (
        output_directory /
        "report.json"
    )

    markdown_file = (
        output_directory /
        "report.md"
    )

    save_json(
        report,
        json_file,
    )

    save_markdown(
        markdown,
        markdown_file,
    )

    return (
        json_file,
        markdown_file,
    )


# ==========================================================
# Export Everything
# ==========================================================

def export_all(
    output_directory: str | Path,
) -> dict:
    """
    Export every artifact needed by the dashboard.
    """

    repository = InsightsRepository()

    output_directory = Path(
        output_directory
    )

    dashboard = export_dashboard(
        repository,
        output_directory,
    )

    topics = export_topics(
        repository,
        output_directory,
    )

    recommendations = (
        export_recommendations(
            repository,
            output_directory,
        )
    )

    report_json, report_md = (
        export_report(
            repository,
            output_directory,
        )
    )

    return {

        "dashboard": dashboard,

        "topics": topics,

        "recommendations":
            recommendations,

        "report_json":
            report_json,

        "report_markdown":
            report_md,

    }


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    files = export_all(
        "artifacts"
    )

    print()

    print("VoxForge AI artifacts generated")

    print("-" * 40)

    for name, path in files.items():

        print(f"{name:<20} {path}")

    print()
