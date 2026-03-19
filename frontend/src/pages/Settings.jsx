import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { testConnection } from "../api/client";
import { useToast } from "../components/shared/Toast";
import { DEFAULT_SETTINGS, loadSettings, saveSettings } from "../utils/settings";

const QUANTIZATION_OPTIONS = [
  { label: "INT8", value: "int8" },
  { label: "INT4", value: "int4" },
  { label: "Mixed", value: "mixed" },
  { label: "QEP", value: "qep" },
];

export default function Settings() {
  const initial = loadSettings();
  const { showSuccess } = useToast();
  const apiKeyRef = useRef(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiBaseUrl, setApiBaseUrl] = useState(initial.apiBaseUrl);
  const [defaultQuantMode, setDefaultQuantMode] = useState(initial.defaultQuantMode);
  const [defaultTargetSize, setDefaultTargetSize] = useState(initial.defaultTargetSize);
  const [connectionMessage, setConnectionMessage] = useState("");
  const [connectionTone, setConnectionTone] = useState("text-text-muted");

  const testConnectionMutation = useMutation({
    mutationFn: () =>
      testConnection(
        apiBaseUrl || DEFAULT_SETTINGS.apiBaseUrl,
        apiKeyRef.current?.value || initial.apiKey || DEFAULT_SETTINGS.apiKey,
      ),
  });

  function handleSave() {
    saveSettings({
      apiBaseUrl,
      apiKey: apiKeyRef.current?.value || DEFAULT_SETTINGS.apiKey,
      defaultQuantMode,
      defaultTargetSize,
    });
    showSuccess("Settings saved");
  }

  async function handleTestConnection() {
    try {
      const stats = await testConnectionMutation.mutateAsync();
      setConnectionTone("text-accent");
      setConnectionMessage(`✓ Connected — ${stats.total_jobs} jobs in database`);
    } catch {
      setConnectionTone("text-accent-danger");
      setConnectionMessage("✗ Cannot reach API");
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="cx-card space-y-5">
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
            SETTINGS
          </div>
          <div className="mt-2 text-lg font-semibold text-text-primary">
            Dashboard Configuration
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <div className="cx-section-label">API Base URL</div>
            <input
              type="text"
              value={apiBaseUrl}
              onChange={(event) => setApiBaseUrl(event.target.value)}
              className="cx-input mt-2"
            />
          </div>

          <div>
            <div className="flex items-center justify-between">
              <div className="cx-section-label">API Key</div>
              <button
                type="button"
                onClick={() => setShowApiKey((current) => !current)}
                className="text-xs uppercase tracking-[0.18em] text-text-muted"
              >
                {showApiKey ? "Hide" : "Show"}
              </button>
            </div>
            <input
              ref={apiKeyRef}
              defaultValue={initial.apiKey}
              type={showApiKey ? "text" : "password"}
              className="cx-input mt-2"
              autoComplete="off"
            />
          </div>

          <div>
            <div className="cx-section-label">Default Quantization Mode</div>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
              {QUANTIZATION_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setDefaultQuantMode(option.value)}
                  className={[
                    "rounded border px-3 py-2 text-sm transition-colors duration-200",
                    defaultQuantMode === option.value
                      ? "border-accent bg-accent text-black"
                      : "border-border-dark bg-bg-secondary text-[#aaaaaa] hover:text-text-primary",
                  ].join(" ")}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="cx-section-label">Default Target Size (GB)</div>
            <input
              type="number"
              value={defaultTargetSize}
              onChange={(event) => setDefaultTargetSize(Number(event.target.value))}
              className="cx-input mt-2"
            />
          </div>

          <button type="button" onClick={handleSave} className="cx-button w-full sm:w-auto">
            Save
          </button>
        </div>
      </div>

      <div className="cx-card space-y-4">
        <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
          CONNECTION TEST
        </div>
        <button
          type="button"
          onClick={handleTestConnection}
          disabled={testConnectionMutation.isPending}
          className="cx-button w-full sm:w-auto"
        >
          {testConnectionMutation.isPending ? "TESTING..." : "TEST CONNECTION"}
        </button>
        {connectionMessage ? (
          <div className={["text-sm", connectionTone].join(" ")}>{connectionMessage}</div>
        ) : null}
      </div>
    </div>
  );
}

Settings.propTypes = {};
