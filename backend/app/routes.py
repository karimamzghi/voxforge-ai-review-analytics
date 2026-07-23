"""
FastAPI routes for the dashboard.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

from .services import (
    ArtifactNotFoundError,
    ArtifactReadError,
    ensure_artifacts,
    get_artifact_status,
    get_dashboard,
    get_dashboard_overview,
    get_markdown_report,
    get_recommendation,
    get_recommendations,
    get_recommendations_by_priority,
    get_report,
    get_topic,
    get_topics,
    get_top_recommendations,
    refresh_artifacts,
    search_topics,
)

router = APIRouter(
    prefix="/api",
    tags=["Insights"],
)


# ==========================================================
# Health
# ==========================================================

@router.get("/health")
def health():
    """
    Health endpoint.
    """

    return {
        "status": "ok",
        "service": "voxforge-ai-review-analytics",
    }


# ==========================================================
# Artifact status
# ==========================================================

@router.get("/artifacts/status")
def artifact_status():
    """
    Show exported artifact status.
    """

    try:
        return get_artifact_status()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.post("/artifacts/refresh")
def refresh():
    """
    Rebuild every exported artifact.
    """

    try:
        return refresh_artifacts()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/artifacts/check")
def check():
    """
    Ensure exported artifacts exist.
    """

    try:
        return ensure_artifacts()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# Dashboard
# ==========================================================

@router.get("/dashboard")
def dashboard():

    try:
        return get_dashboard()

    except (ArtifactNotFoundError, ArtifactReadError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/dashboard/overview")
def overview():

    try:
        return get_dashboard_overview()

    except (ArtifactNotFoundError, ArtifactReadError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# Topics
# ==========================================================

@router.get("/topics")
def topics():

    try:
        return get_topics()

    except (ArtifactNotFoundError, ArtifactReadError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/topics/search")
def topic_search(query: str):

    try:
        return search_topics(query)

    except (ArtifactNotFoundError, ArtifactReadError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/topics/{topic_id}")
def topic(topic_id: int):

    try:

        result = get_topic(topic_id)

        if result is None:

            raise HTTPException(
                status_code=404,
                detail="Topic not found.",
            )

        return result

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# Recommendations
# ==========================================================

@router.get("/recommendations")
def recommendations():

    try:
        return get_recommendations()

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/recommendations/top")
def top_recommendations(limit: int = 5):

    try:
        return get_top_recommendations(limit)

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/recommendations/priority/{priority}")
def recommendation_priority(priority: str):

    try:
        return get_recommendations_by_priority(priority)

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get("/recommendations/{topic_id}")
def recommendation(topic_id: int):

    try:

        result = get_recommendation(topic_id)

        if result is None:

            raise HTTPException(
                status_code=404,
                detail="Recommendation not found.",
            )

        return result

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# Reports
# ==========================================================

@router.get("/report")
def report():

    try:
        return get_report()

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@router.get(
    "/report/markdown",
    response_class=PlainTextResponse,
)
def markdown_report():

    try:
        return get_markdown_report()

    except (ArtifactNotFoundError, ArtifactReadError) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# Root
# ==========================================================

@router.get("/")
def root():

    return {
        "application": "VoxForge AI Review Analytics",
        "status": "running",
        "version": "1.0",
        "documentation": "/docs",
    }