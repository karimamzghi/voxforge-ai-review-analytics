from src.insights.repository import InsightsRepository
from src.insights.recommendation import (
    generate_all_recommendations,
    generate_recommendation_payload,
)

repository = InsightsRepository()

recommendations = generate_all_recommendations(repository)
payload = generate_recommendation_payload(repository)

print(recommendations[:2])
print(payload["summary"])