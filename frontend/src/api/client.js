import axios from "axios";

import { pushToast } from "../components/shared/Toast";

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_API_KEY = "compressx-dev-key";
const OPTIONAL_STATUS_CODES = new Set([404, 405]);
const recentErrorTimestamps = new Map();

function resolveBaseUrl(rawBaseUrl) {
  const normalized = (rawBaseUrl || DEFAULT_BASE_URL).replace(/\/$/, "");

  if (
    import.meta.env.DEV &&
    /^https?:\/\/(localhost|127\.0\.0\.1):8000$/i.test(normalized)
  ) {
    return "/api";
  }

  return normalized;
}

function getStoredValue(key, fallback) {
  if (typeof window === "undefined") {
    return fallback;
  }

  const value = window.localStorage.getItem(key);
  return value === null || value === "" ? fallback : value;
}

function createClient(baseURLOverride, apiKeyOverride) {
  const baseURL = resolveBaseUrl(
    baseURLOverride || getStoredValue("apiBaseUrl", DEFAULT_BASE_URL),
  );
  const apiKey = apiKeyOverride || getStoredValue("apiKey", DEFAULT_API_KEY);

  return axios.create({
    baseURL,
    timeout: 30000,
    headers: {
      "X-API-Key": apiKey,
    },
  });
}

function extractErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  return error.message || "API request failed.";
}

function emitApiError(error) {
  const message = extractErrorMessage(error);
  const now = Date.now();
  const lastSeen = recentErrorTimestamps.get(message) || 0;

  if (now - lastSeen > 5000) {
    recentErrorTimestamps.set(message, now);
    pushToast({ message, tone: "error" });
  }
}

function isOptionalEndpointError(error) {
  const statusCode = error.response?.status;
  const detail = error.response?.data?.detail;

  if (!OPTIONAL_STATUS_CODES.has(statusCode)) {
    return false;
  }

  return statusCode === 405 || detail === "Not Found" || detail === "Method Not Allowed";
}

function buildCompressionPayload(formData) {
  const quantizationMode = formData.quantization_mode === "qep"
    ? "mixed"
    : formData.quantization_mode;

  return {
    model_id: formData.model_id,
    target_size_gb: Number(formData.target_size_gb),
    distill: Boolean(formData.distill),
    domain_data_url: formData.domain_data_url || undefined,
    quantization_mode: quantizationMode || "mixed",
    head_pruning_threshold: Number(
      formData.head_pruning_threshold ?? formData.head_threshold ?? 0.01,
    ),
    layer_redundancy_threshold: Number(
      formData.layer_redundancy_threshold ?? formData.layer_threshold ?? 0.85,
    ),
    aggressive: Boolean(formData.aggressive),
    use_qep: Boolean(formData.use_qep),
    qep_threshold: Number(formData.qep_threshold ?? 0.3),
    calibration_data_path: formData.calibration_data_path || undefined,
    webhook_url: formData.webhook_url || undefined,
    hf_token: formData.hf_token || undefined,
  };
}

function parseFilename(headers, jobId) {
  const disposition = headers["content-disposition"] || "";
  const match = disposition.match(/filename="?([^"]+)"?/i);
  return match ? match[1] : `${jobId}.zip`;
}

export async function submitCompression(formData) {
  try {
    const { data } = await createClient().post(
      "/compress",
      buildCompressionPayload(formData),
    );
    return data;
  } catch (error) {
    emitApiError(error);
    throw error;
  }
}

export async function getJobStatus(jobId) {
  try {
    const { data } = await createClient().get(`/status/${jobId}`);
    return data;
  } catch (error) {
    emitApiError(error);
    throw error;
  }
}

export async function getJobDetail(jobId) {
  try {
    const { data } = await createClient().get(`/jobs/${jobId}`);
    return data;
  } catch (error) {
    if (isOptionalEndpointError(error)) {
      return null;
    }

    emitApiError(error);
    throw error;
  }
}

export async function getRecentJobs(limit = 10) {
  try {
    const { data } = await createClient().get("/jobs", {
      params: { limit },
    });

    return {
      jobs: Array.isArray(data.jobs) ? data.jobs : [],
      supported: true,
    };
  } catch (error) {
    if (isOptionalEndpointError(error)) {
      return {
        jobs: [],
        supported: false,
      };
    }

    emitApiError(error);
    throw error;
  }
}

export async function getStats() {
  try {
    const { data } = await createClient().get("/stats");
    return data;
  } catch (error) {
    emitApiError(error);
    throw error;
  }
}

export async function cancelJob(jobId) {
  try {
    await createClient().delete(`/jobs/${jobId}`);
    return true;
  } catch (error) {
    if (isOptionalEndpointError(error)) {
      return null;
    }

    emitApiError(error);
    throw error;
  }
}

export async function downloadModel(jobId) {
  try {
    const response = await createClient().get(`/download/${jobId}`, {
      responseType: "blob",
    });
    const filename = parseFilename(response.headers, jobId);
    const blobUrl = window.URL.createObjectURL(response.data);
    const anchor = document.createElement("a");
    anchor.href = blobUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.URL.revokeObjectURL(blobUrl);
    return filename;
  } catch (error) {
    emitApiError(error);
    throw error;
  }
}

export async function testConnection(baseURL, apiKey) {
  try {
    const { data } = await createClient(baseURL, apiKey).get("/stats");
    return data;
  } catch (error) {
    emitApiError(error);
    throw error;
  }
}
