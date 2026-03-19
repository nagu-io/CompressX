import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import JobDetail from "../components/jobs/JobDetail";
import { useToast } from "../components/shared/Toast";
import { useDownloadModel, useJob, useJobDetail } from "../hooks/useJob";
import {
  extractRatioValue,
  formatRatio,
  formatSizeGb,
  normalizeStatus,
  safeArray,
} from "../utils/format";

const STATUS_STAGE_MAP = {
  ANALYZING: "sensitivity",
  QUANTIZING: "quantization",
  PRUNING: "pruning",
  DISTILLING: "distillation",
  EVALUATING: "evaluation",
};

const BASE_STAGES = [
  { key: "sensitivity", label: "Sensitivity Analysis" },
  { key: "quantization", label: "Quantization" },
  { key: "pruning", label: "Structural Pruning" },
  { key: "distillation", label: "Distillation" },
  { key: "evaluation", label: "Evaluation" },
];

function stripStagePrefix(label) {
  return String(label || "").replace(/^Stage \d+:\s*/i, "");
}

function inferQepEnabled(detail, statusPayload) {
  if (detail?.report?.qep_applied || Number(detail?.report?.qep_layers_1bit) > 0) {
    return true;
  }

  return safeArray(detail?.logs).some((line) => /qep/i.test(line)) ||
    safeArray(statusPayload?.logs).some((line) => /qep/i.test(line));
}

function buildStages(statusPayload, detail) {
  const backendStages = Array.isArray(detail?.stages) ? detail.stages : [];
  const mappedStages = new Map(
    backendStages.map((stage) => [
      stage.key,
      {
        key: stage.key,
        label: stripStagePrefix(stage.label),
        status: stage.status,
        durationSeconds: stage.duration_seconds ?? null,
        progress: stage.progress ?? 0,
      },
    ]),
  );

  const normalizedStatus = normalizeStatus(statusPayload?.status || detail?.status);
  const currentStageKey = STATUS_STAGE_MAP[statusPayload?.stage] || null;
  const qepEnabled = inferQepEnabled(detail, statusPayload);
  const shouldShowDistillation = detail?.distill || statusPayload?.stage === "DISTILLING";

  const visibleStages = BASE_STAGES.filter(
    (stage) => shouldShowDistillation || stage.key !== "distillation",
  );

  if (qepEnabled) {
    visibleStages.splice(2, 0, { key: "qep", label: "QEP" });
  }

  return visibleStages.map((stage) => {
    if (mappedStages.has(stage.key)) {
      return mappedStages.get(stage.key);
    }

    if (stage.key === "qep") {
      if (detail?.report?.qep_applied) {
        return {
          key: "qep",
          label: "QEP",
          status: "COMPLETE",
          durationSeconds: null,
          progress: 1,
        };
      }

      if (safeArray(detail?.logs).some((line) => /qep/i.test(line))) {
        return {
          key: "qep",
          label: "QEP",
          status: normalizedStatus === "RUNNING" ? "RUNNING" : "WAITING",
          durationSeconds: detail?.stage_elapsed_seconds ?? null,
          progress: statusPayload?.progress ?? 0,
        };
      }
    }

    if (stage.key === currentStageKey && normalizedStatus === "RUNNING") {
      return {
        key: stage.key,
        label: stage.label,
        status: "RUNNING",
        durationSeconds: detail?.stage_elapsed_seconds ?? null,
        progress: statusPayload?.progress ?? 0,
      };
    }

    const currentStageIndex = visibleStages.findIndex((item) => item.key === currentStageKey);
    const stageIndex = visibleStages.findIndex((item) => item.key === stage.key);
    if (currentStageIndex > -1 && stageIndex < currentStageIndex) {
      return {
        key: stage.key,
        label: stage.label,
        status: "COMPLETE",
        durationSeconds: null,
        progress: 1,
      };
    }

    if (normalizedStatus === "DONE") {
      return {
        key: stage.key,
        label: stage.label,
        status: "COMPLETE",
        durationSeconds: null,
        progress: 1,
      };
    }

    if (normalizedStatus === "FAILED" && stage.key === currentStageKey) {
      return {
        key: stage.key,
        label: stage.label,
        status: "FAILED",
        durationSeconds: detail?.stage_elapsed_seconds ?? null,
        progress: statusPayload?.progress ?? 0,
      };
    }

    return {
      key: stage.key,
      label: stage.label,
      status: "WAITING",
      durationSeconds: null,
      progress: 0,
    };
  });
}

function buildSyntheticLogs(statusPayload, detail, metrics) {
  const lines = [];
  const stageName = detail?.stage || statusPayload?.stage;
  if (stageName) {
    lines.push(`Stage: ${stageName}`);
  }
  if (metrics.ratioValue) {
    lines.push(`Compression ratio so far: ${formatRatio(metrics.ratioValue)}`);
  }
  if (metrics.currentSizeGb) {
    lines.push(`Current size: ${formatSizeGb(metrics.currentSizeGb)}`);
  }
  if (statusPayload?.error) {
    lines.push(`Error: ${statusPayload.error}`);
  }
  return lines;
}

function buildResolvedJob(jobId, statusPayload, detail) {
  const normalizedStatus = normalizeStatus(statusPayload?.status || detail?.status);
  const report = detail?.report || {};
  const metrics = detail?.metrics || {};
  const originalSizeGb = metrics.original_size_gb ?? report.original_size_gb ?? null;
  const finalSizeGb = metrics.final_size_gb ?? report.final_size_gb ?? null;
  const currentSizeGb = metrics.current_size_gb ?? statusPayload?.current_size_gb ?? finalSizeGb;
  const ratioValue = metrics.compression_ratio_value ?? extractRatioValue(
    metrics.compression_ratio ?? report.compression_ratio,
    originalSizeGb,
    finalSizeGb ?? currentSizeGb,
  );

  const resolvedMetrics = {
    originalSizeGb,
    currentSizeGb,
    finalSizeGb,
    ratioValue,
    accuracyRetention:
      metrics.accuracy_retention_percent ??
      report.accuracy_retention_percent ??
      statusPayload?.accuracy_retention ??
      null,
    layersPruned: metrics.layers_pruned ?? report.layers_pruned ?? 0,
    headsPruned: metrics.heads_pruned ?? report.heads_pruned ?? 0,
    qepLayers1bit: report.qep_layers_1bit ?? 0,
    stopReason: report.target_outcome ?? null,
    optimizationPasses: report.optimization_passes ?? 0,
    perplexityOriginal: metrics.perplexity_original ?? report.perplexity_original ?? null,
    perplexityCompressed:
      metrics.perplexity_compressed ?? report.perplexity_compressed ?? null,
    speedOriginal: metrics.inference_speed_original ?? report.inference_speed_original ?? null,
    speedCompressed:
      metrics.inference_speed_compressed ?? report.inference_speed_compressed ?? null,
  };

  const logs = safeArray(detail?.logs);

  return {
    jobId,
    modelId: detail?.model_id || statusPayload?.model_id || jobId,
    status: normalizedStatus,
    createdAt: detail?.created_at || statusPayload?.created_at || null,
    updatedAt: detail?.updated_at || statusPayload?.updated_at || null,
    elapsedSeconds: detail?.elapsed_seconds ?? null,
    error: detail?.error || statusPayload?.error || null,
    metrics: resolvedMetrics,
    stages: buildStages(statusPayload, detail),
    logs: logs.length > 0 ? logs : buildSyntheticLogs(statusPayload, detail, resolvedMetrics),
  };
}

function isNotFoundError(error) {
  return error?.response?.status === 404;
}

export default function JobStatus() {
  const { id } = useParams();
  const { showSuccess } = useToast();
  const jobQuery = useJob(id);
  const detailQuery = useJobDetail(id);
  const downloadMutation = useDownloadModel(id);
  const [tick, setTick] = useState(0);

  const resolvedJob = useMemo(() => {
    if (!jobQuery.job) {
      return null;
    }
    return buildResolvedJob(id, jobQuery.job, detailQuery.detail);
  }, [detailQuery.detail, id, jobQuery.job]);

  useEffect(() => {
    if (!resolvedJob || ["DONE", "FAILED", "CANCELLED"].includes(resolvedJob.status)) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setTick((current) => current + 1);
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [resolvedJob]);

  const elapsedSeconds = useMemo(() => {
    if (!resolvedJob) {
      return null;
    }

    if (
      resolvedJob.elapsedSeconds !== null &&
      ["DONE", "FAILED", "CANCELLED"].includes(resolvedJob.status)
    ) {
      return resolvedJob.elapsedSeconds;
    }

    if (!resolvedJob.createdAt) {
      return resolvedJob.elapsedSeconds;
    }

    const createdTime = new Date(resolvedJob.createdAt).getTime();
    if (Number.isNaN(createdTime)) {
      return resolvedJob.elapsedSeconds;
    }

    return Math.max(0, Math.floor((Date.now() - createdTime) / 1000));
  }, [resolvedJob, tick]);

  async function handleDownload() {
    const filename = await downloadMutation.mutateAsync();
    showSuccess(`${filename} download started`);
  }

  if (jobQuery.isLoading && !resolvedJob) {
    return <div className="cx-card h-[420px] animate-pulse" />;
  }

  if (jobQuery.isError && isNotFoundError(jobQuery.error)) {
    return <div className="cx-card text-sm text-text-primary">Job not found.</div>;
  }

  if (!resolvedJob) {
    return (
      <div className="cx-card text-sm text-text-primary">
        Unable to load job status right now.
      </div>
    );
  }

  return (
    <JobDetail
      job={resolvedJob}
      elapsedSeconds={elapsedSeconds}
      isDownloading={downloadMutation.isPending}
      onDownload={handleDownload}
    />
  );
}

JobStatus.propTypes = {};
