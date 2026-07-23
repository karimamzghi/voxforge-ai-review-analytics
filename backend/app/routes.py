from fastapi import APIRouter, HTTPException

from backend.app.config import APP_NAME, APP_VERSION
from backend.app.repository import DashboardRepository
from backend.app.schemas import (
    DashboardData,
    HealthResponse,
    Recommendation,
    Topic,
)

router = APIRouter()
repository = DashboardRepository()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION)


@router.get("/dashboard", response_model=DashboardData)
def dashboard() -> DashboardData:
    return repository.load()


@router.get("/topics", response_model=list[Topic])
def topics() -> list[Topic]:
    return repository.load().topics


@router.get("/topics/{topic_id}", response_model=Topic)
def topic_detail(topic_id: int) -> Topic:
    topic = repository.get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic


@router.get("/recommendations", response_model=list[Recommendation])
def recommendations() -> list[Recommendation]:
    return repository.load().recommendations
