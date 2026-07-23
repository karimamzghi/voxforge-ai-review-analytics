import { useEffect, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { TopicCard } from "../components/TopicCard";
import { api } from "../services/api";
import type { Topic } from "../types/dashboard";

export function TopicsPage() {
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.getTopics().then(setTopics).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <ErrorState message={error} />;
  if (!topics) return <LoadingState />;
  return <div><h1 className="text-3xl font-bold text-white">Topics</h1><p className="mt-2 text-slate-400">Interpretable product themes discovered from review language.</p><div className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-3">{topics.map((topic) => <TopicCard key={topic.id} topic={topic} />)}</div></div>;
}
