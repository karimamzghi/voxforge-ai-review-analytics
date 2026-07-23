import { useEffect, useState } from "react";

import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { RecommendationCard } from "../components/RecommendationCard";
import { api } from "../services/api";
import type { ExecutiveReport } from "../types/dashboard";

type ListSectionProps = {
  title: string;
  items: string[];
};

function ListSection({ title, items }: ListSectionProps) {
  if (items.length === 0) {
    return null;
  }

  return (
    <section className="card p-6">
      <h2 className="text-xl font-semibold text-white">{title}</h2>
      <ul className="mt-4 space-y-3">
        {items.map((item) => (
          <li
            key={item}
            className="rounded-xl border border-line bg-black/10 p-4 text-sm leading-6 text-slate-300"
          >
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function ReportPage() {
  const [report, setReport] = useState<ExecutiveReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getReport()
      .then(setReport)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) {
    return <ErrorState message={error} />;
  }

  if (!report) {
    return <LoadingState message="Loading executive report…" />;
  }

  return (
    <div className="space-y-6">
      <header className="card p-7 sm:p-10">
        <p className="text-sm uppercase tracking-[0.22em] text-forgeOrange">
          Executive report
        </p>
        <h1 className="mt-3 text-4xl font-black text-white">
          Customer intelligence summary
        </h1>
        <p className="mt-8 text-lg leading-8 text-slate-300">
          {report.executive_summary}
        </p>

        {report.generated_at && (
          <p className="mt-6 text-xs text-slate-500">
            Generated: {new Date(report.generated_at).toLocaleString()}
          </p>
        )}
      </header>

      <div className="grid gap-6 xl:grid-cols-2">
        <ListSection title="Key findings" items={report.key_findings} />
        <ListSection title="Business risks" items={report.business_risks} />
      </div>

      <ListSection
        title="Business opportunities"
        items={report.business_opportunities}
      />

      {report.recommendations.length > 0 && (
        <section>
          <h2 className="text-2xl font-bold text-white">
            Recommended actions
          </h2>
          <div className="mt-4 space-y-4">
            {report.recommendations.map((recommendation) => (
              <RecommendationCard
                key={`${recommendation.id}-${recommendation.topic}`}
                recommendation={recommendation}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
