import { MetricCard } from "./MetricCard";
import type { Metric } from "../types/dashboard";

type Props = {
  metrics: Metric[];
};

export function DashboardOverview({ metrics }: Props) {
  return (
    <section>
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.22em] text-forgeBlue">
          Customer intelligence
        </p>
        <h2 className="mt-2 text-2xl font-bold text-white">
          Analytics overview
        </h2>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
    </section>
  );
}
