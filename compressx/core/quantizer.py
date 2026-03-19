from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from compressx.exceptions import ConfigurationError
from compressx.utils.io import directory_size_gb, read_json
from compressx.utils.quantization import (
    apply_quantization_plan_inplace,
    estimate_quantized_size_gb,
)

GPTQ_SENSITIVE_THRESHOLD = 0.7


def _is_test_model_source(source_model: Any | None, model_source: str) -> bool:
    module_name = getattr(getattr(source_model, "__class__", None), "__module__", "")
    return module_name.startswith("compressx.tests.") or model_source == "toy/model"


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _load_gptq_symbols() -> tuple[Any, Any]:
    """Load GPTQ symbols lazily so CPU-only installs still import cleanly."""

    transformers = importlib.import_module("transformers")
    return transformers.GPTQConfig, transformers.AutoModelForCausalLM


def _load_sensitivity_report(
    output_dir: Path,
    fallback_report: dict[str, float] | None = None,
) -> dict[str, float]:
    """Load the persisted sensitivity report when available."""

    report_path = output_dir / "sensitivity_report.json"
    if report_path.exists():
        payload = read_json(report_path)
        return {str(key): float(value) for key, value in payload.items()}
    return {
        str(key): float(value)
        for key, value in (fallback_report or {}).items()
    }


def _build_bits_map(
    sensitivity_report: dict[str, float],
    *,
    sensitive_threshold: float = GPTQ_SENSITIVE_THRESHOLD,
    default_bits: int = 4,
    sensitive_bits: int = 8,
) -> dict[str, int]:
    """Map sensitivity scores to the GPTQ bit-width plan."""

    return {
        layer_name: sensitive_bits if score > sensitive_threshold else default_bits
        for layer_name, score in sensitivity_report.items()
    }


def _resolve_architecture_name(source_model: Any | None) -> str:
    """Return the configured architecture name when available."""

    config = getattr(source_model, "config", None)
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return source_model.__class__.__name__ if source_model is not None else "unknown"


def _candidate_batch_sizes(calibration_batch_size: int) -> list[int]:
    """Build the GPTQ calibration retry sequence."""

    candidates: list[int] = []
    for batch_size in (max(1, calibration_batch_size), 2, 1):
        if batch_size not in candidates:
            candidates.append(batch_size)
    return candidates


def _find_saved_model_artifact(output_dir: Path) -> Path:
    """Locate the primary model artifact written by save_pretrained."""

    for pattern in ("*.safetensors", "*.bin"):
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return output_dir


def build_quantization_plan(
    layer_names: list[str],
    sensitivity_report: dict[str, float],
    *,
    threshold: float,
    default_bits: int,
    sensitive_bits: int,
) -> dict[str, int]:
    if not sensitivity_report:
        return {layer_name: default_bits for layer_name in layer_names}

    return {
        layer_name: sensitive_bits
        if sensitivity_report.get(layer_name, 0.0) > threshold
        else default_bits
        for layer_name in layer_names
    }


def _run_cpu_path(
    *,
    model_source: str,
    source_model: Any | None = None,
    tokenizer: Any,
    output_dir,
    offload_dir,
    calibration_data: list[str],
    calibration_source_name: str,
    calibration_batch_size: int,
    quantization_plan: dict[str, int],
    default_bits: int,
    gptq_group_size: int,
    gptq_desc_act: bool,
    gptq_damp_percent: float,
    trust_remote_code: bool,
    hf_token: str | None,
    stage_logger: Any,
) -> dict[str, Any]:
    effective_batch_size = max(1, calibration_batch_size)
    del (
        tokenizer,
        output_dir,
        offload_dir,
        calibration_data,
        calibration_source_name,
        gptq_group_size,
        gptq_desc_act,
        gptq_damp_percent,
        trust_remote_code,
        hf_token,
    )

    if source_model is None or not hasattr(source_model, "state_dict"):
        raise ConfigurationError(
            "Manual quantization requires a loaded torch model instance."
        )

    warnings: list[str] = []
    if _is_test_model_source(source_model, model_source):
        warnings.append(
            "Using the in-memory quantization test backend for the toy model."
        )

    if default_bits < 4:
        warnings.append(
            f"Aggressive {default_bits}-bit quantization is enabled. Expect a larger "
            "accuracy drop than the default mixed 4/8-bit profile."
        )

    apply_quantization_plan_inplace(source_model, quantization_plan)
    estimated_size_gb = estimate_quantized_size_gb(
        source_model.state_dict(),
        quantization_plan,
    )
    stage_logger.info(
        "Applied manual %s-bit quantization plan in-memory. Estimated packed size: %.4f GB.",
        default_bits,
        estimated_size_gb,
    )
    return {
        "model": source_model,
        "method": "manual_cpu",
        "saved_path": None,
        "final_size_gb": estimated_size_gb,
        "effective_batch_size": effective_batch_size,
        "warnings": warnings,
        "quantization_plan": quantization_plan,
        "cuda_available": False,
    }


def _run_gptq_path(
    *,
    model_source: str,
    source_model: Any | None = None,
    tokenizer: Any,
    output_dir,
    offload_dir,
    calibration_data: list[str],
    calibration_source_name: str,
    calibration_batch_size: int,
    quantization_plan: dict[str, int],
    default_bits: int,
    gptq_group_size: int,
    gptq_desc_act: bool,
    gptq_damp_percent: float,
    trust_remote_code: bool,
    hf_token: str | None,
    stage_logger: Any,
    sensitivity_report: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the CUDA GPTQ backend and fall back to CPU when unsupported."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload_dir = Path(offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)

    try:
        GPTQConfig, AutoModelForCausalLM = _load_gptq_symbols()
    except ImportError:
        stage_logger.warning(
            "GPTQConfig not available. Install: pip install transformers>=4.35.0 optimum auto-gptq"
        )
        return _run_cpu_path(
            model_source=model_source,
            source_model=source_model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            offload_dir=offload_dir,
            calibration_data=calibration_data,
            calibration_source_name=calibration_source_name,
            calibration_batch_size=calibration_batch_size,
            quantization_plan=quantization_plan,
            default_bits=default_bits,
            gptq_group_size=gptq_group_size,
            gptq_desc_act=gptq_desc_act,
            gptq_damp_percent=gptq_damp_percent,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            stage_logger=stage_logger,
        )

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    bits_map = _build_bits_map(
        _load_sensitivity_report(output_dir, sensitivity_report),
    )
    dataset: str | list[str] = (
        calibration_data if calibration_source_name != "wikitext2" and calibration_data else "wikitext2"
    )
    load_kwargs: dict[str, Any] = {
        "quantization_config": None,
        "device_map": "auto",
        "offload_folder": str(offload_dir),
        "trust_remote_code": trust_remote_code,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    for batch_size in _candidate_batch_sizes(calibration_batch_size):
        quantization_config = GPTQConfig(
            bits=4,
            dataset=dataset,
            tokenizer=tokenizer,
            group_size=gptq_group_size,
            desc_act=gptq_desc_act,
            damp_percent=gptq_damp_percent,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
        )
        load_kwargs["quantization_config"] = quantization_config

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **load_kwargs,
            )
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            saved_path = _find_saved_model_artifact(output_dir)
            final_size_gb = directory_size_gb(output_dir)
            warnings: list[str] = []
            if any(bits == 8 for bits in bits_map.values()):
                warnings.append(
                    "Sensitive layers were identified for 8-bit handling; the GPTQ backend currently applies the base 4-bit configuration globally."
                )
            return {
                "model": model,
                "method": "gptq_cuda",
                "saved_path": saved_path,
                "final_size_gb": final_size_gb,
                "effective_batch_size": batch_size,
                "warnings": warnings,
                "quantization_plan": bits_map or quantization_plan,
                "cuda_available": True,
            }
        except RuntimeError as exc:
            if is_oom_error(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size > 1:
                    next_batch_size = max(1, batch_size // 2)
                    stage_logger.warning(
                        "CUDA OOM during GPTQ calibration with batch_size=%s. Retrying with batch_size=%s.",
                        batch_size,
                        next_batch_size,
                    )
                    continue
                stage_logger.warning(
                    "CUDA OOM during GPTQ calibration with batch_size=1. Falling back to the CPU quantization path."
                )
                return _run_cpu_path(
                    model_source=model_source,
                    source_model=source_model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    offload_dir=offload_dir,
                    calibration_data=calibration_data,
                    calibration_source_name=calibration_source_name,
                    calibration_batch_size=calibration_batch_size,
                    quantization_plan=quantization_plan,
                    default_bits=default_bits,
                    gptq_group_size=gptq_group_size,
                    gptq_desc_act=gptq_desc_act,
                    gptq_damp_percent=gptq_damp_percent,
                    trust_remote_code=trust_remote_code,
                    hf_token=hf_token,
                    stage_logger=stage_logger,
                )
            raise
        except ValueError:
            architecture_name = _resolve_architecture_name(source_model)
            stage_logger.warning(
                "Architecture %s not supported by GPTQ, using CPU path",
                architecture_name,
            )
            return _run_cpu_path(
                model_source=model_source,
                source_model=source_model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                offload_dir=offload_dir,
                calibration_data=calibration_data,
                calibration_source_name=calibration_source_name,
                calibration_batch_size=calibration_batch_size,
                quantization_plan=quantization_plan,
                default_bits=default_bits,
                gptq_group_size=gptq_group_size,
                gptq_desc_act=gptq_desc_act,
                gptq_damp_percent=gptq_damp_percent,
                trust_remote_code=trust_remote_code,
                hf_token=hf_token,
                stage_logger=stage_logger,
            )

    return _run_cpu_path(
        model_source=model_source,
        source_model=source_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        offload_dir=offload_dir,
        calibration_data=calibration_data,
        calibration_source_name=calibration_source_name,
        calibration_batch_size=calibration_batch_size,
        quantization_plan=quantization_plan,
        default_bits=default_bits,
        gptq_group_size=gptq_group_size,
        gptq_desc_act=gptq_desc_act,
        gptq_damp_percent=gptq_damp_percent,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
        stage_logger=stage_logger,
    )


def quantize_model(
    *,
    model_source: str,
    source_model: Any | None = None,
    tokenizer: Any,
    output_dir,
    offload_dir,
    calibration_data: list[str],
    calibration_source_name: str,
    calibration_batch_size: int,
    quantization_plan: dict[str, int],
    default_bits: int,
    gptq_group_size: int,
    gptq_desc_act: bool,
    gptq_damp_percent: float,
    trust_remote_code: bool,
    hf_token: str | None,
    stage_logger: Any,
) -> dict[str, Any]:
    """Backward-compatible CPU quantization entry point."""

    return _run_cpu_path(
        model_source=model_source,
        source_model=source_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        offload_dir=offload_dir,
        calibration_data=calibration_data,
        calibration_source_name=calibration_source_name,
        calibration_batch_size=calibration_batch_size,
        quantization_plan=quantization_plan,
        default_bits=default_bits,
        gptq_group_size=gptq_group_size,
        gptq_desc_act=gptq_desc_act,
        gptq_damp_percent=gptq_damp_percent,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
        stage_logger=stage_logger,
    )


class MixedPrecisionQuantizer:
    """Compatibility entry point that preserves the stage-style core API."""

    def run(self, context: Any) -> None:
        from compressx.stages.quantization import MixedPrecisionQuantizerStage

        MixedPrecisionQuantizerStage().run(context)


__all__ = [
    "MixedPrecisionQuantizer",
    "GPTQ_SENSITIVE_THRESHOLD",
    "_build_bits_map",
    "_run_cpu_path",
    "_run_gptq_path",
    "build_quantization_plan",
    "is_oom_error",
    "quantize_model",
]
