import type {
  DashboardData,
  ExecutiveReport,
  Metric,
  Recommendation,
  SentimentBreakdown,
  Topic,
} from "../types/dashboard";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

type JsonObject = Record<string, unknown>;

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      Accept: "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let message = `API request failed with status ${response.status}.`;

    try {
      const payload = (await response.json()) as {
        detail?: string;
        message?: string;
      };
      message = payload.detail ?? payload.message ?? message;
    } catch {
      // Keep the HTTP fallback message for non-JSON responses.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

function asObject(value: unknown): JsonObject {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asPercent(value: unknown): number {
  const numeric = asNumber(value);
  return numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric;
}

function firstNumber(
  source: JsonObject,
  keys: string[],
  fallback = 0,
): number {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) {
      return asNumber(source[key], fallback);
    }
  }
  return fallback;
}

function firstString(
  source: JsonObject,
  keys: string[],
  fallback = "",
): string {
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return fallback;
}

function normalizeSentiment(value: unknown): SentimentBreakdown {
  const source = asObject(value);

  return {
    positive: asPercent(
      source.positive_percent ??
        source.positive_percentage ??
        source.positive_rate ??
        source.positive_ratio ??
        source.positive,
    ),
    neutral: asPercent(
      source.neutral_percent ??
        source.neutral_percentage ??
        source.neutral_rate ??
        source.neutral_ratio ??
        source.neutral,
    ),
    negative: asPercent(
      source.negative_percent ??
        source.negative_percentage ??
        source.negative_rate ??
        source.negative_ratio ??
        source.negative,
    ),
    dominant: firstString(source, ["dominant", "overall_sentiment"]),
    health: firstString(source, ["health", "sentiment_health"]),
  };
}

function normalizeReviews(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }

  const source = asObject(value);

  return [
    ...asArray(source.positive),
    ...asArray(source.negative),
    ...asArray(source.neutral),
  ].filter((item): item is string => typeof item === "string");
}

function normalizeTopic(value: unknown, index: number): Topic {
  const source = asObject(value);
  const sentiment = normalizeSentiment(source.sentiment ?? source);
  const reviewCount = firstNumber(
    source,
    ["review_count", "topic_size", "size"],
  );

  return {
    id: firstNumber(source, ["id", "topic_id"], index),
    name: firstString(
      source,
      ["name", "topic_name", "topic"],
      `Topic ${index + 1}`,
    ),
    description: firstString(
      source,
      ["description", "summary", "narrative"],
      "No topic summary is available.",
    ),
    review_count: reviewCount,
    review_share_percent: asPercent(
      source.review_share_percent ?? source.review_share,
    ),
    sentiment,
    keywords: asArray(
      source.keywords ?? source.top_keywords ?? source.terms,
    ).filter((item): item is string => typeof item === "string"),
    representative_reviews: normalizeReviews(
      source.representative_reviews ?? source.sample_reviews,
    ),
    recommendation: firstString(
      source,
      ["recommendation", "suggested_action", "action"],
      "Continue monitoring this topic and review representative feedback.",
    ),
  };
}

function normalizeRecommendation(
  value: unknown,
  index: number,
): Recommendation {
  const source = asObject(value);

  return {
    id: firstNumber(source, ["id", "rank", "topic_id"], index + 1),
    title: firstString(
      source,
      ["title", "recommendation"],
      "Review this customer-feedback theme",
    ),
    priority: firstString(source, ["priority"], "Medium"),
    topic: firstString(
      source,
      ["topic", "topic_name"],
      "Customer feedback",
    ),
    evidence: firstString(
      source,
      ["evidence", "rationale", "business_impact"],
      "This recommendation was generated from topic and sentiment analysis.",
    ),
    action: firstString(
      source,
      ["action", "suggested_action", "recommendation"],
      "Review the supporting customer feedback and define an owner.",
    ),
    severity:
      source.severity === undefined
        ? undefined
        : asNumber(source.severity),
  };
}

function normalizeRecommendations(value: unknown): Recommendation[] {
  if (Array.isArray(value)) {
    return value.map(normalizeRecommendation);
  }

  const source = asObject(value);
  const items =
    source.recommendations ??
    source.top_recommendations ??
    source.items ??
    [];

  return asArray(items).map(normalizeRecommendation);
}

function normalizeMetrics(
  value: unknown,
  overview: JsonObject,
  topicCount: number,
): Metric[] {
  const explicitMetrics = asArray(value);

  if (explicitMetrics.length > 0) {
    return explicitMetrics.map((item) => {
      const metric = asObject(item);
      return {
        label: firstString(metric, ["label"], "Metric"),
        value: firstString(
          metric,
          ["value"],
          String(firstNumber(metric, ["numeric_value"])),
        ),
        detail: firstString(metric, ["detail", "description"]),
      };
    });
  }

  const totalReviews = firstNumber(overview, [
    "total_reviews",
    "review_count",
  ]);
  const positive = asPercent(
    overview.positive_percent ??
      overview.positive_percentage ??
      overview.positive_rate,
  );
  const negative = asPercent(
    overview.negative_percent ??
      overview.negative_percentage ??
      overview.negative_rate,
  );

  return [
    {
      label: "Reviews analysed",
      value: totalReviews.toLocaleString(),
      detail: "Customer reviews processed by the analytics pipeline",
    },
    {
      label: "Topics discovered",
      value: String(
        firstNumber(overview, ["total_topics"], topicCount),
      ),
      detail: "Interpretable themes discovered through clustering",
    },
    {
      label: "Positive sentiment",
      value: `${positive.toFixed(1)}%`,
      detail: "Reviews classified as positive",
    },
    {
      label: "Negative sentiment",
      value: `${negative.toFixed(1)}%`,
      detail: "Reviews classified as negative",
    },
  ];
}

function normalizeDashboard(value: unknown): DashboardData {
  const source = asObject(value);
  const overview = asObject(source.overview);
  const topics = asArray(source.topics).map(normalizeTopic);
  const recommendations = normalizeRecommendations(source.recommendations);

  const overallSentiment =
    Object.keys(overview).length > 0
      ? normalizeSentiment(overview)
      : normalizeSentiment(source.overall_sentiment);

  return {
    metrics: normalizeMetrics(source.metrics, overview, topics.length),
    overall_sentiment: overallSentiment,
    topics,
    recommendations,
    executive_summary: firstString(
      source,
      ["executive_summary", "summary"],
      firstString(
        asObject(source.report),
        ["executive_summary", "summary"],
        firstString(overview, ["summary"]),
      ),
    ),
  };
}

function normalizeReport(value: unknown): ExecutiveReport {
  const source = asObject(value);

  return {
    generated_at: firstString(source, ["generated_at"]),
    executive_summary: firstString(
      source,
      ["executive_summary", "summary"],
      "No executive summary is available.",
    ),
    key_findings: asArray(source.key_findings).filter(
      (item): item is string => typeof item === "string",
    ),
    business_risks: asArray(
      source.business_risks ?? source.risks,
    ).filter((item): item is string => typeof item === "string"),
    business_opportunities: asArray(
      source.business_opportunities ?? source.opportunities,
    ).filter((item): item is string => typeof item === "string"),
    recommendations: normalizeRecommendations(source.recommendations),
  };
}

export const api = {
  getHealth: () =>
    request<{ status: string; service?: string }>("/api/health"),

  getDashboard: async (): Promise<DashboardData> =>
    normalizeDashboard(await request<unknown>("/api/dashboard")),

  getTopics: async (): Promise<Topic[]> =>
    asArray(await request<unknown>("/api/topics")).map(normalizeTopic),

  getTopic: async (id: number): Promise<Topic> => {
    try {
      return normalizeTopic(
        await request<unknown>(`/api/topics/${id}`),
        id,
      );
    } catch (error) {
      const topics = await api.getTopics();
      const topic = topics.find((item) => item.id === id);

      if (!topic) {
        throw error;
      }

      return topic;
    }
  },

  getRecommendations: async (): Promise<Recommendation[]> =>
    normalizeRecommendations(
      await request<unknown>("/api/recommendations"),
    ),

  getReport: async (): Promise<ExecutiveReport> =>
    normalizeReport(await request<unknown>("/api/report")),

  refreshArtifacts: () =>
    request<Record<string, unknown>>("/api/artifacts/refresh", {
      method: "POST",
    }),
};
