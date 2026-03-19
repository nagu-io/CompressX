import { useMutation, useQuery } from "@tanstack/react-query";

import {
  cancelJob,
  downloadModel,
  getJobDetail,
  getJobStatus,
} from "../api/client";

function getPollingInterval(jobData) {
  const status = jobData?.status;
  return status === "DONE" || status === "FAILED" || status === "CANCELLED"
    ? false
    : 3000;
}

export function useJob(jobId) {
  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (queryContext) => getPollingInterval(queryContext.state.data),
  });

  return {
    ...query,
    job: query.data ?? null,
  };
}

export function useJobDetail(jobId) {
  const query = useQuery({
    queryKey: ["job-detail", jobId],
    queryFn: () => getJobDetail(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (queryContext) => (
      queryContext.state.data === null
        ? false
        : getPollingInterval(queryContext.state.data)
    ),
  });

  return {
    ...query,
    detail: query.data ?? null,
  };
}

export function useDownloadModel(jobId) {
  return useMutation({
    mutationFn: () => downloadModel(jobId),
  });
}

export function useCancelJob(jobId) {
  return useMutation({
    mutationFn: () => cancelJob(jobId),
  });
}
