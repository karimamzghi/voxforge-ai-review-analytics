import { useEffect, useState } from "react";

import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { RecommendationCard } from "../components/RecommendationCard";
import { api } from "../services/api";
import type { Recommendation } from "../types/dashboard";

export function RecommendationsPage() {
  const [items, setItems] = useState<Recommendation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getRecommendations()
      .then(setItems)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) {
    return <ErrorState message={error} />;
  }

  if (!items) {
    return <LoadingState message="Loading recommendations…" />;
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-white">Recommendations</h1>
      <p className="mt-2 text-slate-400">
        Evidence-based actions derived from topic and sentiment analysis.
      </p>

      {items.length === 0 ? (
        <p className="card mt-8 p-6 text-slate-400">
          No recommendations are currently available.
        </p>
      ) : (
        <div className="mt-8 space-y-4">
          {items.map((item) => (
            <RecommendationCard
              key={`${item.id}-${item.topic}`}
              recommendation={item}
            />
          ))}
        </div>
      )}
    </div>
  );
}
