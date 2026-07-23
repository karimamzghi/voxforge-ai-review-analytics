import { useEffect, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { MetricCard } from "../components/MetricCard";
import { TopicCard } from "../components/TopicCard";
import { api } from "../services/api";
import type { DashboardData } from "../types/dashboard";

const chartColors = ["#34d399", "#fbbf24", "#fb7185"];

export function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDashboard().then(setData).catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data) return <LoadingState />;

  const sentimentData = [
    { name: "Positive", value: data.overall_sentiment.positive },
    { name: "Neutral", value: data.overall_sentiment.neutral },
    { name: "Negative", value: data.overall_sentiment.negative },
  ];

  return (
    <div className="space-y-8">
      <header className="overflow-hidden rounded-3xl border border-line bg-gradient-to-br from-forgeBlue/15 via-panel to-forgeOrange/10 p-7 shadow-glow sm:p-10">
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-forgeBlue">AI-powered customer intelligence</p>
        <h1 className="mt-4 max-w-3xl text-4xl font-black tracking-tight text-white sm:text-6xl">
          Turn customer voices into <span className="gradient-text">product decisions.</span>
        </h1>
        <p className="mt-5 max-w-2xl text-base leading-7 text-slate-300">
          VoxForge AI combines DistilBERT sentiment analysis, semantic topic discovery and business intelligence to surface what customers value—and what needs attention.
        </p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {data.metrics.map((metric) => <MetricCard key={metric.label} metric={metric} />)}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_1.9fr]">
        <article className="card p-6">
          <h2 className="text-lg font-semibold text-white">Overall sentiment</h2>
          <p className="mt-1 text-sm text-slate-500">Distribution across all analysed reviews</p>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={sentimentData} dataKey="value" nameKey="name" innerRadius={66} outerRadius={102} paddingAngle={3}>
                  {sentimentData.map((entry, index) => <Cell key={entry.name} fill={chartColors[index]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: "#101827", border: "1px solid #23314d", borderRadius: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {sentimentData.map((item, index) => (
              <div key={item.name} className="rounded-xl bg-white/[0.03] p-3">
                <span className="mx-auto mb-2 block h-2 w-2 rounded-full" style={{ backgroundColor: chartColors[index] }} />
                <p className="text-slate-500">{item.name}</p><p className="mt-1 font-semibold text-white">{item.value}%</p>
              </div>
            ))}
          </div>
        </article>

        <article className="card p-6">
          <div className="flex items-end justify-between gap-4">
            <div><h2 className="text-lg font-semibold text-white">Priority insight</h2><p className="mt-1 text-sm text-slate-500">Highest-impact opportunity from Phase 3</p></div>
            <span className="rounded-full bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-300">High priority</span>
          </div>
          <div className="mt-6 rounded-2xl border border-forgeOrange/30 bg-forgeOrange/5 p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-forgeOrange">Batteries</p>
            <h3 className="mt-3 text-2xl font-bold text-white">Investigate reliability and shelf life</h3>
            <p className="mt-3 leading-7 text-slate-300">Battery reviews contain the highest negative sentiment rate at 12.3%, well above the product-level baseline.</p>
            <p className="mt-5 text-sm text-slate-400"><span className="font-semibold text-white">Recommended action:</span> review supplier quality, storage conditions and performance expectations.</p>
          </div>
          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            {data.topics.slice(2, 5).map((topic) => (
              <div key={topic.id} className="rounded-xl border border-line bg-black/10 p-4">
                <p className="truncate text-sm font-medium text-white">{topic.name}</p>
                <p className="mt-2 text-2xl font-bold text-emerald-300">{topic.sentiment.positive}%</p>
                <p className="text-xs text-slate-500">positive sentiment</p>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section>
        <div className="mb-4"><h2 className="text-2xl font-bold text-white">Customer topics</h2><p className="mt-1 text-sm text-slate-500">Explore the six themes discovered by TF-IDF, SVD and K-Means.</p></div>
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {data.topics.map((topic) => <TopicCard key={topic.id} topic={topic} />)}
        </div>
      </section>
    </div>
  );
}
