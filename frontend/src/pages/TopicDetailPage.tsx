import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { api } from "../services/api";
import type { Topic } from "../types/dashboard";

export function TopicDetailPage() {
  const { id } = useParams();
  const topicId = Number(id);
  const [topic, setTopic] = useState<Topic | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(topicId)) {
      setError("The topic identifier is invalid.");
      return;
    }

    api
      .getTopic(topicId)
      .then(setTopic)
      .catch((requestError: Error) => setError(requestError.message));
  }, [topicId]);

  if (error) {
    return <ErrorState message={error} />;
  }

  if (!topic) {
    return <LoadingState message="Loading topic details…" />;
  }

  return (
    <div className="space-y-6">
      <header className="card p-8">
        <p className="text-sm uppercase tracking-[0.2em] text-forgeBlue">
          Topic {topic.id + 1}
        </p>
        <h1 className="mt-3 text-4xl font-black text-white">{topic.name}</h1>
        <p className="mt-3 max-w-3xl text-slate-400">
          {topic.description}
        </p>
      </header>

      <section className="grid gap-5 md:grid-cols-4">
        {[
          ["Reviews", topic.review_count.toLocaleString()],
          ["Dataset share", `${topic.review_share_percent.toFixed(1)}%`],
          ["Positive", `${topic.sentiment.positive.toFixed(1)}%`],
          ["Negative", `${topic.sentiment.negative.toFixed(1)}%`],
        ].map(([label, value]) => (
          <div key={label} className="card p-5">
            <p className="text-sm text-slate-500">{label}</p>
            <p className="mt-2 text-3xl font-bold text-white">{value}</p>
          </div>
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <article className="card p-6">
          <h2 className="text-xl font-semibold text-white">Top keywords</h2>
          <div className="mt-5 flex flex-wrap gap-3">
            {topic.keywords.length > 0 ? (
              topic.keywords.map((word) => (
                <span
                  key={word}
                  className="rounded-full border border-forgeBlue/30 bg-forgeBlue/10 px-4 py-2 text-sm text-cyan-100"
                >
                  {word}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-500">
                No keywords were exported for this topic.
              </p>
            )}
          </div>
        </article>

        <article className="card p-6">
          <h2 className="text-xl font-semibold text-white">
            Business recommendation
          </h2>
          <p className="mt-4 leading-7 text-slate-300">
            {topic.recommendation}
          </p>
        </article>
      </section>

      <article className="card p-6">
        <h2 className="text-xl font-semibold text-white">
          Representative reviews
        </h2>
        <div className="mt-5 space-y-3">
          {topic.representative_reviews.length > 0 ? (
            topic.representative_reviews.map((review, index) => (
              <blockquote
                key={`${index}-${review.slice(0, 30)}`}
                className="rounded-xl border border-line bg-black/10 p-4 text-sm italic leading-6 text-slate-300"
              >
                “{review}”
              </blockquote>
            ))
          ) : (
            <p className="text-sm text-slate-500">
              No representative reviews were exported for this topic.
            </p>
          )}
        </div>
      </article>
    </div>
  );
}
