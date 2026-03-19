import { useQuery } from "@tanstack/react-query";

import { getRecentJobs, getStats } from "../api/client";

export function useStats(options = {}) {
  const query = useQuery({
    queryKey: ["stats"],
    queryFn: getStats,
    refetchInterval: 30000,
    ...options,
  });

  return {
    ...query,
    stats: query.data ?? null,
  };
}

export function useRecentJobs(limit = 10, options = {}) {
  const query = useQuery({
    queryKey: ["recent-jobs", limit],
    queryFn: () => getRecentJobs(limit),
    refetchInterval: 30000,
    ...options,
  });

  return {
    ...query,
    jobs: query.data?.jobs ?? [],
    supported: query.data?.supported ?? true,
  };
}
