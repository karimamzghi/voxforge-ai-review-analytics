import type { DashboardData, Recommendation, Topic } from "../types/dashboard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getDashboard: () => request<DashboardData>("/api/dashboard"),
  getTopics: () => request<Topic[]>("/api/topics"),
  getTopic: (id: number) => request<Topic>(`/api/topics/${id}`),
  getRecommendations: () => request<Recommendation[]>("/api/recommendations"),
};
