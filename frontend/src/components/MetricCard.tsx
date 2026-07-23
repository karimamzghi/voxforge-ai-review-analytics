import type { Metric } from "../types/dashboard";

type Props = { metric: Metric };

export function MetricCard({ metric }: Props) {
  return (
    <article className="card p-5 transition duration-200 hover:-translate-y-1 hover:border-forgeBlue/50">
      <p className="text-sm text-slate-400">{metric.label}</p>
      <p className="mt-2 text-3xl font-bold text-white">{metric.value}</p>
      <p className="mt-2 text-xs text-slate-500">{metric.detail}</p>
    </article>
  );
}
