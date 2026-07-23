export type Priority = "High" | "Medium" | "Low" | string;

export type SentimentBreakdown = {
  positive: number;
  neutral: number;
  negative: number;
  dominant?: string;
  health?: string;
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
  review_share_percent: number;
  sentiment: SentimentBreakdown;
  keywords: string[];
  representative_reviews: string[];
  recommendation: string;
};

export type Recommendation = {
  id: number;
  title: string;
  priority: Priority;
  topic: string;
  evidence: string;
  action: string;
  severity?: number;
};

export type DashboardData = {
  metrics: Metric[];
  overall_sentiment: SentimentBreakdown;
  topics: Topic[];
  recommendations: Recommendation[];
  executive_summary: string;
};

export type ExecutiveReport = {
  generated_at?: string;
  executive_summary: string;
  key_findings: string[];
  business_risks: string[];
  business_opportunities: string[];
  recommendations: Recommendation[];
};
