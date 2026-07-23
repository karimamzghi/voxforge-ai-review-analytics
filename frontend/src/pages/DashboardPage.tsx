import { DashboardOverview } from "../components/DashboardOverview";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { RecommendationCard } from "../components/RecommendationCard";
import { SentimentChart } from "../components/SentimentChart";
import { TopicCard } from "../components/TopicCard";
import { useDashboard } from "../hooks/useDashboard";

export function DashboardPage() {
  const { data, loading, error, reload } = useDashboard();

  if (loading) {
    return <LoadingState message="Loading dashboard…" />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => void reload()} />;
  }

  if (!data) {
    return (
      <ErrorState
        message="The API returned no dashboard data."
        onRetry={() => void reload()}
      />
    );
  }

  return (
    <div className="space-y-10">
      <header>
        <p className="text-sm uppercase tracking-[0.22em] text-forgeOrange">
          VoxForge AI
        </p>
        <h1 className="mt-3 text-4xl font-black text-white sm:text-5xl">
          Customer review intelligence
        </h1>
        <p className="mt-4 max-w-3xl leading-7 text-slate-400">
          Sentiment analysis, topic discovery and actionable recommendations
          generated from customer reviews.
        </p>
      </header>

      <DashboardOverview metrics={data.metrics} />

      <section className="grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
        <SentimentChart sentiment={data.overall_sentiment} />

        <article className="card p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-forgeOrange">
            Executive summary
          </p>
          <h2 className="mt-2 text-2xl font-bold text-white">
            What the feedback tells us
          </h2>
          <p className="mt-5 leading-7 text-slate-300">
            {data.executive_summary ||
              "No executive summary is currently available."}
          </p>
        </article>
      </section>

      {data.recommendations.length > 0 && (
        <section>
          <div className="mb-4">
            <h2 className="text-2xl font-bold text-white">
              Priority recommendation
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              The highest-impact action identified by the analytics pipeline.
            </p>
          </div>

          <RecommendationCard recommendation={data.recommendations[0]} />
        </section>
      )}

      <section>
        <div className="mb-4">
          <h2 className="text-2xl font-bold text-white">Customer topics</h2>
          <p className="mt-1 text-sm text-slate-500">
            Themes discovered through clustering and enriched with sentiment.
          </p>
        </div>

        {data.topics.length === 0 ? (
          <p className="card p-6 text-slate-400">
            No topics are available.
          </p>
        ) : (
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {data.topics.map((topic) => (
              <TopicCard key={topic.id} topic={topic} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
