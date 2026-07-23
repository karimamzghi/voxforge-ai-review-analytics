import type { Recommendation } from "../types/dashboard";

type Props = {
  recommendation: Recommendation;
};

function priorityClass(priority: string): string {
  const normalized = priority.toLowerCase();

  if (normalized === "high") {
    return "border-red-400/30 bg-red-500/10 text-red-200";
  }

  if (normalized === "low") {
    return "border-emerald-400/30 bg-emerald-500/10 text-emerald-200";
  }

  return "border-forgeOrange/30 bg-forgeOrange/10 text-orange-200";
}

export function RecommendationCard({ recommendation }: Props) {
  return (
    <article className="card p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-forgeBlue">
            {recommendation.topic}
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">
            {recommendation.title}
          </h2>
        </div>

        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${priorityClass(
            recommendation.priority,
          )}`}
        >
          {recommendation.priority}
        </span>
      </div>

      <p className="mt-4 text-sm leading-6 text-slate-400">
        <strong className="text-slate-200">Evidence:</strong>{" "}
        {recommendation.evidence}
      </p>

      <p className="mt-3 text-sm leading-6 text-slate-300">
        <strong className="text-white">Recommended action:</strong>{" "}
        {recommendation.action}
      </p>

      {recommendation.severity !== undefined && (
        <p className="mt-4 text-xs text-slate-500">
          Severity score: {recommendation.severity.toFixed(2)}
        </p>
      )}
    </article>
  );
}
