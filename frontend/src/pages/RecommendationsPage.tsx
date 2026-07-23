import { useEffect, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { api } from "../services/api";
import type { Recommendation } from "../types/dashboard";

export function RecommendationsPage() {
  const [items, setItems] = useState<Recommendation[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.getRecommendations().then(setItems).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <ErrorState message={error} />;
  if (!items) return <LoadingState />;
  return <div><h1 className="text-3xl font-bold text-white">Recommendations</h1><p className="mt-2 text-slate-400">Evidence-based actions derived from topic and sentiment analysis.</p><div className="mt-8 space-y-4">{items.map((item) => <article key={item.id} className="card p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-xs uppercase tracking-[0.18em] text-forgeBlue">{item.topic}</p><h2 className="mt-2 text-xl font-semibold text-white">{item.title}</h2></div><span className="rounded-full border border-forgeOrange/30 bg-forgeOrange/10 px-3 py-1 text-xs text-orange-200">{item.priority}</span></div><p className="mt-4 text-sm text-slate-400"><strong className="text-slate-200">Evidence:</strong> {item.evidence}</p><p className="mt-2 text-sm text-slate-300"><strong className="text-white">Action:</strong> {item.action}</p></article>)}</div></div>;
}
