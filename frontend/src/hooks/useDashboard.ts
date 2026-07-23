import { useCallback, useEffect, useState } from "react";

import { api } from "../services/api";
import type { DashboardData } from "../types/dashboard";

type UseDashboardResult = {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
};

export function useDashboard(): UseDashboardResult {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      setData(await api.getDashboard());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to load dashboard data.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { data, loading, error, reload };
}
