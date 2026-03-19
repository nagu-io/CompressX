export const DEFAULT_SETTINGS = {
  apiBaseUrl: "http://localhost:8000",
  apiKey: "compressx-dev-key",
  defaultQuantMode: "mixed",
  defaultTargetSize: 3.0,
};

const STORAGE_KEYS = {
  apiBaseUrl: "apiBaseUrl",
  apiKey: "apiKey",
  defaultQuantMode: "defaultQuantMode",
  defaultTargetSize: "defaultTargetSize",
};

function readStorage(key, fallback) {
  if (typeof window === "undefined") {
    return fallback;
  }

  const value = window.localStorage.getItem(key);
  return value === null || value === "" ? fallback : value;
}

export function loadSettings() {
  const targetSize = Number(
    readStorage(
      STORAGE_KEYS.defaultTargetSize,
      String(DEFAULT_SETTINGS.defaultTargetSize),
    ),
  );

  return {
    apiBaseUrl: readStorage(STORAGE_KEYS.apiBaseUrl, DEFAULT_SETTINGS.apiBaseUrl),
    apiKey: readStorage(STORAGE_KEYS.apiKey, DEFAULT_SETTINGS.apiKey),
    defaultQuantMode: readStorage(
      STORAGE_KEYS.defaultQuantMode,
      DEFAULT_SETTINGS.defaultQuantMode,
    ),
    defaultTargetSize: Number.isFinite(targetSize)
      ? targetSize
      : DEFAULT_SETTINGS.defaultTargetSize,
  };
}

export function saveSettings(settings) {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(
    STORAGE_KEYS.apiBaseUrl,
    settings.apiBaseUrl || DEFAULT_SETTINGS.apiBaseUrl,
  );
  window.localStorage.setItem(
    STORAGE_KEYS.apiKey,
    settings.apiKey || DEFAULT_SETTINGS.apiKey,
  );
  window.localStorage.setItem(
    STORAGE_KEYS.defaultQuantMode,
    settings.defaultQuantMode || DEFAULT_SETTINGS.defaultQuantMode,
  );
  window.localStorage.setItem(
    STORAGE_KEYS.defaultTargetSize,
    String(settings.defaultTargetSize || DEFAULT_SETTINGS.defaultTargetSize),
  );
}

function yamlString(value) {
  if (value === null || value === undefined || value === "") {
    return "null";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (typeof value === "number") {
    return String(value);
  }

  return `"${String(value).replace(/"/g, '\\"')}"`;
}

export function buildYamlPreview(values) {
  const useQep = values.quantizationMode === "qep";
  const quantMode = useQep ? "mixed" : values.quantizationMode;

  return [
    `model_id: ${yamlString(values.modelId || "facebook/opt-125m")}`,
    `target_size_gb: ${yamlString(values.targetSize)}`,
    "quantization:",
    `  mode: ${yamlString(quantMode)}`,
    `  head_threshold: ${yamlString(values.headThreshold)}`,
    `  layer_threshold: ${yamlString(values.layerThreshold)}`,
    "distillation:",
    `  enabled: ${yamlString(values.distill)}`,
    `  domain_data_url: ${yamlString(values.domainDataUrl)}`,
    "qep:",
    `  enabled: ${yamlString(useQep)}`,
    `  threshold: ${yamlString(0.3)}`,
    `aggressive: ${yamlString(values.aggressive)}`,
    `calibration_data_path: ${yamlString(values.calibrationDataPath)}`,
    `webhook_url: ${yamlString(values.webhookUrl)}`,
  ].join("\n");
}
