import { ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";
import type { Topic } from "../types/dashboard";

type Props = { topic: Topic };

export function TopicCard({ topic }: Props) {
  return (
    <article className="card flex h-full flex-col p-5 transition hover:-translate-y-1 hover:border-forgeOrange/50">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-forgeBlue">Topic {topic.id + 1}</p>
          <h3 className="mt-2 text-xl font-semibold text-white">{topic.name}</h3>
        </div>
        <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300">
          {topic.review_count.toLocaleString()} reviews
        </span>
      </div>
      <p className="mt-3 flex-1 text-sm leading-6 text-slate-400">{topic.description}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        {topic.keywords.slice(0, 4).map((keyword) => (
          <span key={keyword} className="rounded-full border border-line bg-black/20 px-3 py-1 text-xs text-slate-300">
            {keyword}
          </span>
        ))}
      </div>
      <div className="mt-5 flex items-center justify-between border-t border-line/70 pt-4">
        <span className="text-sm text-emerald-300">{topic.sentiment.positive}% positive</span>
        <Link className="inline-flex items-center gap-1 text-sm font-medium text-forgeOrange" to={`/topics/${topic.id}`}>
          View details <ArrowUpRight size={15} />
        </Link>
      </div>
    </article>
  );
}
