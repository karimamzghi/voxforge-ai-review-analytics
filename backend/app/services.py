"""
This module is responsible for:
- loading exported insight artifacts
- validating that files exist
- returning dashboard, topic, recommendation, and report data
- refreshing artifacts from the analytics package when requested

The API routes should call these service functions instead of reading
JSON files directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.insights.exporter import export_all


# ==========================================================
# Paths
# ==========================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"

DASHBOARD_FILE = DATA_DIR / "dashboard.json"
TOPICS_FILE = DATA_DIR / "topics.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"
REPORT_FILE = DATA_DIR / "report.json"
REPORT_MARKDOWN_FILE = DATA_DIR / "report.md"


# ==========================================================
# Exceptions
# ==========================================================

class ArtifactNotFoundError(FileNotFoundError):
    """
    Raised when an expected exported artifact does not exist.
    """


class ArtifactReadError(RuntimeError):
    """
    Raised when an artifact exists but cannot be read or parsed.
    """


# ==========================================================
# Generic file readers
# ==========================================================

def _require_file(path: Path) -> Path:
    """
    Ensure that an artifact exists and is a regular file.

    Parameters
    ----------
    path:
        Path to the expected artifact.

    Returns
    -------
    Path
        The validated file path.

    Raises
    ------
    ArtifactNotFoundError
        If the file does not exist.
    """

    if not path.exists():
        raise ArtifactNotFoundError(
            f"Artifact not found: {path}. "
            "Run the insights exporter before starting the API."
        )

    if not path.is_file():
        raise ArtifactNotFoundError(
            f"Expected a file but found something else: {path}"
        )

    return path


def load_json_artifact(path: str | Path) -> Any:
    """
    Load and parse a JSON artifact.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON content.

    Raises
    ------
    ArtifactNotFoundError
        If the file does not exist.

    ArtifactReadError
        If the file cannot be opened or contains invalid JSON.
    """

    artifact_path = _require_file(Path(path))

    try:
        with artifact_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ArtifactReadError(
            f"Invalid JSON in artifact {artifact_path}: {error}"
        ) from error

    except OSError as error:
        raise ArtifactReadError(
            f"Could not read artifact {artifact_path}: {error}"
        ) from error


def load_text_artifact(path: str | Path) -> str:
    """
    Load a text-based artifact such as a Markdown report.
    """

    artifact_path = _require_file(Path(path))

    try:
        return artifact_path.read_text(
            encoding="utf-8",
        )

    except OSError as error:
        raise ArtifactReadError(
            f"Could not read artifact {artifact_path}: {error}"
        ) from error


# ==========================================================
# Dashboard service
# ==========================================================

def get_dashboard() -> dict[str, Any]:
    """
    Return the complete dashboard payload.
    """

    payload = load_json_artifact(DASHBOARD_FILE)

    if not isinstance(payload, dict):
        raise ArtifactReadError(
            "dashboard.json must contain a JSON object."
        )

    return payload


def get_dashboard_overview() -> dict[str, Any]:
    """
    Return only the dashboard overview section.
    """

    dashboard = get_dashboard()
    overview = dashboard.get("overview", {})

    if not isinstance(overview, dict):
        raise ArtifactReadError(
            "The dashboard overview must be a JSON object."
        )

    return overview


# ==========================================================
# Topic services
# ==========================================================

def get_topics() -> list[dict[str, Any]]:
    """
    Return all generated topic summaries.
    """

    topics = load_json_artifact(TOPICS_FILE)

    if not isinstance(topics, list):
        raise ArtifactReadError(
            "topics.json must contain a JSON list."
        )

    return topics


def get_topic(topic_id: int | str) -> dict[str, Any] | None:
    """
    Return one topic by its topic identifier.

    The comparison is performed as a string so that integer and string
    identifiers are both supported.
    """

    requested_id = str(topic_id)

    for topic in get_topics():
        current_id = topic.get("topic_id")

        if str(current_id) == requested_id:
            return topic

    return None


def search_topics(query: str) -> list[dict[str, Any]]:
    """
    Search topic summaries by topic name or summary text.

    The search is case-insensitive.
    """

    cleaned_query = query.strip().lower()

    if not cleaned_query:
        return get_topics()

    matches: list[dict[str, Any]] = []

    for topic in get_topics():
        searchable_values = [
            topic.get("topic", ""),
            topic.get("summary", ""),
            topic.get("narrative", ""),
            topic.get("sentiment_health", ""),
        ]

        searchable_text = " ".join(
            str(value)
            for value in searchable_values
            if value is not None
        ).lower()

        if cleaned_query in searchable_text:
            matches.append(topic)

    return matches


# ==========================================================
# Recommendation services
# ==========================================================

def get_recommendation_payload() -> Any:
    """
    Return the complete recommendation artifact.

    Depending on the exporter implementation, this may be either:
    - a list of recommendations
    - a dictionary containing recommendations and summary metrics
    """

    return load_json_artifact(RECOMMENDATIONS_FILE)


def get_recommendations() -> list[dict[str, Any]]:
    """
    Return the recommendation list from the exported payload.
    """

    payload = get_recommendation_payload()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        recommendations = payload.get("recommendations", [])

        if not isinstance(recommendations, list):
            raise ArtifactReadError(
                "The recommendations field must be a JSON list."
            )

        return recommendations

    raise ArtifactReadError(
        "recommendations.json must contain a list or JSON object."
    )


def get_recommendation(
    topic_id: int | str,
) -> dict[str, Any] | None:
    """
    Return the recommendation associated with one topic.
    """

    requested_id = str(topic_id)

    for recommendation in get_recommendations():
        current_id = recommendation.get("topic_id")

        if str(current_id) == requested_id:
            return recommendation

    return None


def get_recommendations_by_priority(
    priority: str,
) -> list[dict[str, Any]]:
    """
    Filter recommendations by priority.

    Examples
    --------
    High
    Medium
    Low
    """

    requested_priority = priority.strip().lower()

    return [
        recommendation
        for recommendation in get_recommendations()
        if str(
            recommendation.get("priority", "")
        ).strip().lower() == requested_priority
    ]


def get_top_recommendations(
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Return the first ranked recommendations.

    The recommendation exporter is expected to order recommendations by
    priority or rank. A defensive rank sort is applied when possible.
    """

    if limit < 1:
        return []

    recommendations = get_recommendations()

    def ranking_value(
        recommendation: dict[str, Any],
    ) -> tuple[int, float]:
        rank = recommendation.get("rank")
        severity = recommendation.get("severity", 0)

        try:
            numeric_rank = int(rank)
        except (TypeError, ValueError):
            numeric_rank = 10**9

        try:
            numeric_severity = float(severity)
        except (TypeError, ValueError):
            numeric_severity = 0.0

        return numeric_rank, -numeric_severity

    ordered = sorted(
        recommendations,
        key=ranking_value,
    )

    return ordered[:limit]


# ==========================================================
# Report services
# ==========================================================

def get_report() -> dict[str, Any]:
    """
    Return the complete JSON executive report.
    """

    report = load_json_artifact(REPORT_FILE)

    if not isinstance(report, dict):
        raise ArtifactReadError(
            "report.json must contain a JSON object."
        )

    return report


def get_markdown_report() -> str:
    """
    Return the Markdown executive report.
    """

    return load_text_artifact(REPORT_MARKDOWN_FILE)


# ==========================================================
# Artifact metadata
# ==========================================================

def get_artifact_status() -> dict[str, dict[str, Any]]:
    """
    Return availability and metadata for exported artifacts.
    """

    artifact_paths = {
        "dashboard": DASHBOARD_FILE,
        "topics": TOPICS_FILE,
        "recommendations": RECOMMENDATIONS_FILE,
        "report_json": REPORT_FILE,
        "report_markdown": REPORT_MARKDOWN_FILE,
    }

    status: dict[str, dict[str, Any]] = {}

    for name, path in artifact_paths.items():
        exists = path.exists() and path.is_file()

        status[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": (
                path.stat().st_size
                if exists
                else None
            ),
            "modified_at": (
                path.stat().st_mtime
                if exists
                else None
            ),
        }

    return status


def artifacts_are_available() -> bool:
    """
    Return True when all expected artifacts exist.
    """

    status = get_artifact_status()

    return all(
        artifact["exists"]
        for artifact in status.values()
    )


# ==========================================================
# Artifact refresh
# ==========================================================

def refresh_artifacts() -> dict[str, str]:
    """
    Regenerate all exported insight artifacts.

    This executes the analytics exporter and writes the generated files
    into backend/data.

    Returns
    -------
    dict[str, str]
        Mapping of artifact names to generated paths.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = export_all(
        DATA_DIR,
    )

    return {
        name: str(path)
        for name, path in generated_files.items()
    }


def ensure_artifacts() -> dict[str, dict[str, Any]]:
    """
    Ensure that all required artifacts are available.

    Missing artifacts are generated automatically.
    """

    if not artifacts_are_available():
        refresh_artifacts()

    return get_artifact_status()