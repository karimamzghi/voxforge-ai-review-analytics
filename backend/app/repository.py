import json
from functools import lru_cache

from backend.app.config import DASHBOARD_DATA_PATH
from backend.app.schemas import DashboardData, Topic

# Read-only repository for deployment-ready dashboard analytics
class DashboardRepository:
    @staticmethod
    @lru_cache(maxsize=1)
    def load() -> DashboardData:
        if not DASHBOARD_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dashboard data was not found at {DASHBOARD_DATA_PATH}."
            )

        with DASHBOARD_DATA_PATH.open("r", encoding="utf-8") as file:
            return DashboardData.model_validate(json.load(file))

    def get_topic(self, topic_id: int) -> Topic | None:
        return next(
            (topic for topic in self.load().topics if topic.id == topic_id),
            None,
        )
