import { useEffect, useState } from "react";
import { LoadingState } from "../components/LoadingState";
import { api } from "../services/api";

export function ExecutiveReportPage() {
  const [summary, setSummary] = useState<string | null>(null);
  useEffect(() => { api.getDashboard().then((data) => setSummary(data.executive_summary)); }, []);
  if (!summary) return <LoadingState />;
  return <article className="mx-auto max-w-4xl card p-7 sm:p-10"><p className="text-sm uppercase tracking-[0.22em] text-forgeOrange">Executive report</p><h1 className="mt-3 text-4xl font-black text-white">Customer intelligence summary</h1><p className="mt-8 text-lg leading-8 text-slate-300">{summary}</p><div className="mt-10 border-t border-line pt-6 text-sm text-slate-500">Generated from sentiment predictions, topic clusters and representative customer reviews.</div></article>;
}
