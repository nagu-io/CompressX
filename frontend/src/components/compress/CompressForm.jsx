import PropTypes from "prop-types";

const QUANTIZATION_OPTIONS = [
  { label: "INT8", value: "int8" },
  { label: "INT4", value: "int4" },
  { label: "Mixed", value: "mixed" },
  { label: "QEP", value: "qep" },
];

function FieldError({ message }) {
  return message ? <div className="mt-2 text-sm text-accent-danger">{message}</div> : null;
}

FieldError.propTypes = {
  message: PropTypes.string,
};

function buildSliderStyle(value) {
  const percentage = ((Number(value) - 0.5) / 49.5) * 100;
  return {
    background: `linear-gradient(90deg, #00ff88 ${percentage}%, #1e1e1e ${percentage}%)`,
  };
}

export default function CompressForm({
  values,
  errors,
  onChange,
  onSubmit,
  isSubmitting,
}) {
  return (
    <div className="cx-card space-y-6">
      <div>
        <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
          NEW COMPRESSION JOB
        </div>
        <div className="mt-2 text-lg font-semibold text-text-primary">
          Compression Request
        </div>
      </div>

      <div className="space-y-5">
        <div>
          <div className="cx-section-label">Model ID</div>
          <input
            type="text"
            value={values.modelId}
            onChange={(event) => onChange("modelId", event.target.value)}
            placeholder="meta-llama/Meta-Llama-3-8B"
            className="cx-input mt-2"
          />
          <FieldError message={errors.modelId} />
        </div>

        <div>
          <div className="cx-section-label">HuggingFace Token</div>
          <input
            type="password"
            value={values.hfToken}
            onChange={(event) => onChange("hfToken", event.target.value)}
            placeholder="hf_... (required for gated models)"
            className="cx-input mt-2"
          />
        </div>

        <div>
          <div className="flex items-center justify-between">
            <div className="cx-section-label">Target Size</div>
            <div className="font-mono text-sm text-accent">
              {Number(values.targetSize).toFixed(1)} GB
            </div>
          </div>
          <input
            type="range"
            min="0.5"
            max="50"
            step="0.5"
            value={values.targetSize}
            onChange={(event) => onChange("targetSize", Number(event.target.value))}
            style={buildSliderStyle(values.targetSize)}
            className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-full bg-[#1e1e1e]"
          />
        </div>

        <div>
          <div className="cx-section-label">Quantization Mode</div>
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
            {QUANTIZATION_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => onChange("quantizationMode", option.value)}
                className={[
                  "rounded border px-3 py-2 text-sm transition-colors duration-200",
                  values.quantizationMode === option.value
                    ? "border-accent bg-accent text-black"
                    : "border-border-dark bg-bg-secondary text-[#aaaaaa] hover:text-text-primary",
                ].join(" ")}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded border border-border-dark bg-bg-secondary px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="cx-section-label">Enable Distillation</div>
              <div className="mt-1 text-sm text-text-muted">
                Recover quality with domain-specific data after compression.
              </div>
            </div>
            <button
              type="button"
              onClick={() => onChange("distill", !values.distill)}
              className={[
                "relative h-7 w-14 rounded-full transition-colors duration-200",
                values.distill ? "bg-accent" : "bg-[#333333]",
              ].join(" ")}
            >
              <span
                className={[
                  "absolute top-1 h-5 w-5 rounded-full bg-black transition-transform duration-200",
                  values.distill ? "translate-x-8" : "translate-x-1",
                ].join(" ")}
              />
            </button>
          </div>
        </div>

        {values.distill ? (
          <div>
            <div className="cx-section-label">Domain Data URL</div>
            <input
              type="text"
              value={values.domainDataUrl}
              onChange={(event) => onChange("domainDataUrl", event.target.value)}
              placeholder="https://... or local path"
              className="cx-input mt-2"
            />
            <FieldError message={errors.domainDataUrl} />
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <div className="cx-section-label">Head Pruning Threshold</div>
            <input
              type="number"
              step="0.01"
              value={values.headThreshold}
              onChange={(event) => onChange("headThreshold", event.target.value)}
              className="cx-input mt-2"
            />
          </div>
          <div>
            <div className="cx-section-label">Layer Redundancy Threshold</div>
            <input
              type="number"
              step="0.05"
              value={values.layerThreshold}
              onChange={(event) => onChange("layerThreshold", event.target.value)}
              className="cx-input mt-2"
            />
          </div>
        </div>

        <label className="flex items-center gap-3 rounded border border-border-dark bg-bg-secondary px-4 py-3 text-sm text-text-primary">
          <input
            type="checkbox"
            checked={values.aggressive}
            onChange={(event) => onChange("aggressive", event.target.checked)}
            className="h-4 w-4 rounded border-border-dark bg-bg-secondary accent-accent"
          />
          <span>Aggressive mode (more passes, lower accuracy floor)</span>
        </label>

        <div>
          <div className="cx-section-label">Calibration Data Path</div>
          <input
            type="text"
            value={values.calibrationDataPath}
            onChange={(event) => onChange("calibrationDataPath", event.target.value)}
            placeholder="C:\\datasets\\calibration.txt"
            className="cx-input mt-2"
          />
        </div>

        <div>
          <div className="cx-section-label">Webhook URL</div>
          <input
            type="text"
            value={values.webhookUrl}
            onChange={(event) => onChange("webhookUrl", event.target.value)}
            placeholder="https://your-server.com/webhook (optional)"
            className="cx-input mt-2"
          />
          <FieldError message={errors.webhookUrl} />
        </div>

        <button
          type="button"
          onClick={onSubmit}
          disabled={isSubmitting}
          className="cx-button h-12 w-full"
        >
          {isSubmitting ? <span className="spinner-arc h-4 w-4" /> : null}
          {isSubmitting ? "SUBMITTING..." : "START COMPRESSION"}
        </button>
      </div>
    </div>
  );
}

CompressForm.propTypes = {
  values: PropTypes.shape({
    modelId: PropTypes.string.isRequired,
    hfToken: PropTypes.string.isRequired,
    targetSize: PropTypes.number.isRequired,
    quantizationMode: PropTypes.string.isRequired,
    distill: PropTypes.bool.isRequired,
    domainDataUrl: PropTypes.string.isRequired,
    headThreshold: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    layerThreshold: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    aggressive: PropTypes.bool.isRequired,
    calibrationDataPath: PropTypes.string.isRequired,
    webhookUrl: PropTypes.string.isRequired,
  }).isRequired,
  errors: PropTypes.objectOf(PropTypes.string).isRequired,
  onChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSubmitting: PropTypes.bool.isRequired,
};
