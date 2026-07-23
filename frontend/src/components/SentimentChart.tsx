import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import type { SentimentBreakdown } from "../types/dashboard";

type Props = {
  sentiment: SentimentBreakdown;
};

const chartColors = ["#34d399", "#fbbf24", "#fb7185"];

export function SentimentChart({ sentiment }: Props) {
  const data = [
    { name: "Positive", value: sentiment.positive },
    { name: "Neutral", value: sentiment.neutral },
    { name: "Negative", value: sentiment.negative },
  ];

  return (
    <article className="card p-6">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-forgeBlue">
          Sentiment analysis
        </p>
        <h2 className="mt-2 text-lg font-semibold text-white">
          Overall distribution
        </h2>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={58}
              outerRadius={88}
              paddingAngle={3}
            >
              {data.map((item, index) => (
                <Cell
                  key={item.name}
                  fill={chartColors[index]}
                  stroke="transparent"
                />
              ))}
            </Pie>
            <Tooltip
              formatter={(value) => [`${Number(value).toFixed(1)}%`, "Share"]}
              contentStyle={{
                background: "#111827",
                border: "1px solid #243047",
                borderRadius: "12px",
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        {data.map((item, index) => (
          <div
            key={item.name}
            className="rounded-xl bg-white/[0.03] p-3"
          >
            <span
              className="mx-auto mb-2 block h-2 w-2 rounded-full"
              style={{ backgroundColor: chartColors[index] }}
            />
            <p className="text-slate-500">{item.name}</p>
            <p className="mt-1 font-semibold text-white">
              {item.value.toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </article>
  );
}
