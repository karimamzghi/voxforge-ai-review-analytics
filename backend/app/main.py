"""
main.py

FastAPI application entry point for VoxForge AI Review Analytics.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .services import ensure_artifacts


# ==========================================================
# Configuration
# ==========================================================

APP_NAME = "VoxForge AI Review Analytics"
APP_VERSION = "1.0.0"

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

#   Read allowed frontend origins from the environment.
def get_allowed_origins() -> list[str]:
    configured_origins = os.getenv(
        "CORS_FRONTENT_ORIGINS",
        "",
    ).strip()

    if not configured_origins:
        return DEFAULT_ALLOWED_ORIGINS

    return [
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ]


# ==========================================================
# Application lifecycle
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Validate or generate insight artifacts when the API starts.

    Startup should not crash if artifact generation fails. The route
    layer will still expose the error when an artifact is requested.
    """

    try:
        artifact_status = ensure_artifacts()

        app.state.artifact_status = artifact_status
        app.state.artifact_startup_error = None

    except Exception as error:
        app.state.artifact_status = {}
        app.state.artifact_startup_error = str(error)

        print(
            "Warning: insight artifacts could not be prepared "
            f"during startup: {error}"
        )

    yield


# ==========================================================
# FastAPI application
# ==========================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API for customer-review sentiment, topic analysis, "
        "business recommendations and executive reporting."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Routes
# ==========================================================

app.include_router(router)


# ==========================================================
# Non-API root endpoint
# ==========================================================

@app.get(
    "/",
    tags=["Application"],
)
def application_root():
    """
    Root endpoint outside the /api router.
    """

    startup_error = getattr(
        app.state,
        "artifact_startup_error",
        None,
    )

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": (
            "running"
            if startup_error is None
            else "running_with_artifact_error"
        ),
        "api": "/api",
        "health": "/api/health",
        "documentation": "/docs",
        "artifact_startup_error": startup_error,
    }