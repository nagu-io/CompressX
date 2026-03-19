const ACTIVE_STATUSES = new Set([
  "ANALYZING",
  "QUANTIZING",
  "PRUNING",
  "DISTILLING",
  "EVALUATING",
  "RUNNING",
]);

export function normalizeStatus(status) {
  if (!status) {
    return "QUEUED";
  }

  if (ACTIVE_STATUSES.has(status)) {
    return "RUNNING";
  }

  if (status === "DONE" || status === "FAILED" || status === "QUEUED" || status === "CANCELLED") {
    return status;
  }

  return status;
}

export function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function formatSizeGb(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "—";
  }

  if (numeric >= 1024) {
    return `${(numeric / 1024).toFixed(2)} TB`;
  }
  if (numeric >= 10) {
    return `${numeric.toFixed(1)} GB`;
  }
  if (numeric >= 1) {
    return `${numeric.toFixed(2)} GB`;
  }
  return `${numeric.toFixed(3)} GB`;
}

export function formatRatio(value) {
  const numeric = numberOrNull(value);
  return numeric === null ? "—" : `${numeric.toFixed(1)}x`;
}

export function formatPercent(value) {
  const numeric = numberOrNull(value);
  return numeric === null ? "—" : `${numeric.toFixed(1)}%`;
}

export function formatDateTime(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatClock(value = new Date()) {
  return value.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function hasTimestampPrefix(line) {
  return /^\[(\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2}:\d{2}\]/.test(line);
}

export function formatDuration(seconds) {
  const numeric = numberOrNull(seconds);
  if (numeric === null) {
    return "—";
  }

  const total = Math.max(0, Math.floor(numeric));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainingSeconds = total % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

export function accuracyColorClass(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "text-text-muted";
  }
  if (numeric > 90) {
    return "text-accent";
  }
  if (numeric > 80) {
    return "text-accent-warn";
  }
  return "text-accent-danger";
}

export function extractRatioValue(ratioText, originalSize, finalSize) {
  const fromText = typeof ratioText === "string" ? Number.parseFloat(ratioText) : NaN;
  if (Number.isFinite(fromText)) {
    return fromText;
  }

  const original = numberOrNull(originalSize);
  const final = numberOrNull(finalSize);
  if (original === null || final === null || final <= 0) {
    return null;
  }

  return original / final;
}

export function safeArray(value) {
  return Array.isArray(value) ? value : [];
}
