export type SentimentBreakdown = {
  positive: number;
  neutral: number;
  negative: number;
};

export type Metric = {
  label: string;
  value: string;
  detail: string;
};

export type Topic = {
  id: number;
  name: string;
  description: string;
  review_count: number;
  sentiment: SentimentBreakdown;
  keywords: string[];
  representative_reviews: string[];
  recommendation: string;
};

export type Recommendation = {
  id: number;
  title: string;
  priority: "High" | "Medium" | "Low" | string;
  topic: string;
  evidence: string;
  action: string;
};

export type DashboardData = {
  metrics: Metric[];
  overall_sentiment: SentimentBreakdown;
  topics: Topic[];
  recommendations: Recommendation[];
  executive_summary: string;
};
