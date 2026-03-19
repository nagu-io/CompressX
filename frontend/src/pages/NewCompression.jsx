import { useState } from "react";
import { useNavigate } from "react-router-dom";

import CompressForm from "../components/compress/CompressForm";
import ConfigEditor from "../components/compress/ConfigEditor";
import { useToast } from "../components/shared/Toast";
import { useCompress } from "../hooks/useCompress";
import { buildYamlPreview, loadSettings } from "../utils/settings";

function isValidUrl(value) {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
}

function buildInitialValues() {
  const settings = loadSettings();
  return {
    modelId: "",
    hfToken: "",
    targetSize: settings.defaultTargetSize,
    quantizationMode: settings.defaultQuantMode,
    distill: false,
    domainDataUrl: "",
    headThreshold: "0.01",
    layerThreshold: "0.85",
    aggressive: false,
    calibrationDataPath: "",
    webhookUrl: "",
  };
}

export default function NewCompression() {
  const navigate = useNavigate();
  const { showSuccess } = useToast();
  const { submit, isSubmitting } = useCompress();
  const [values, setValues] = useState(buildInitialValues);
  const [errors, setErrors] = useState({});

  function updateField(field, value) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => {
      if (!current[field]) {
        return current;
      }

      const nextErrors = { ...current };
      delete nextErrors[field];
      return nextErrors;
    });
  }

  function validateValues() {
    const nextErrors = {};

    if (!values.modelId.trim()) {
      nextErrors.modelId = "Model ID is required.";
    }

    if (values.distill && !values.domainDataUrl.trim()) {
      nextErrors.domainDataUrl = "Domain data URL is required when distillation is enabled.";
    }

    if (values.webhookUrl && !isValidUrl(values.webhookUrl)) {
      nextErrors.webhookUrl = "Webhook URL must be a valid URL.";
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  }

  async function handleSubmit() {
    if (!validateValues()) {
      return;
    }

    const response = await submit({
      model_id: values.modelId.trim(),
      target_size_gb: values.targetSize,
      distill: values.distill,
      domain_data_url: values.domainDataUrl.trim() || undefined,
      quantization_mode: values.quantizationMode,
      head_threshold: values.headThreshold,
      layer_threshold: values.layerThreshold,
      aggressive: values.aggressive,
      use_qep: values.quantizationMode === "qep",
      qep_threshold: 0.3,
      calibration_data_path: values.calibrationDataPath.trim() || undefined,
      webhook_url: values.webhookUrl.trim() || undefined,
      hf_token: values.hfToken.trim() || undefined,
    });

    showSuccess("Job submitted successfully");
    navigate(`/jobs/${response.job_id}`);
  }

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.5fr_1fr]">
      <CompressForm
        values={values}
        errors={errors}
        onChange={updateField}
        onSubmit={handleSubmit}
        isSubmitting={isSubmitting}
      />
      <ConfigEditor yaml={buildYamlPreview(values)} />
    </div>
  );
}

NewCompression.propTypes = {};
