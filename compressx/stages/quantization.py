from __future__ import annotations

import torch

from compressx.context import CompressionContext
from compressx.core.quantizer import (
    GPTQ_SENSITIVE_THRESHOLD,
    _build_bits_map,
    _run_cpu_path as _run_cpu_backend,
    _run_gptq_path as _run_gptq_backend,
    build_quantization_plan,
)
from compressx.datasets import load_text_samples
from compressx.logging_utils import get_stage_logger
from compressx.stages.base import PipelineStage
from compressx.utils.io import write_json
from compressx.utils.models import find_transformer_layers, module_size_gb


def _plan_threshold(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    clipped_fraction = min(max(fraction, 0.0), 1.0)
    index = min(len(ordered) - 1, max(0, int(clipped_fraction * (len(ordered) - 1))))
    return ordered[index]


def _projected_layer_size_gb(layer: torch.nn.Module, bits: int) -> float:
    total_bytes = 0.0
    for parameter in layer.parameters():
        if parameter.dtype in (torch.float16, torch.float32, torch.bfloat16):
            total_bytes += parameter.numel() * (bits / 8.0)
        else:
            total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes / (1024**3)


def _load_calibration_texts(
    context: CompressionContext,
    stage_logger,
) -> list[str]:
    """Load and validate calibration texts shared by both quantization backends."""

    requested_samples = max(
        context.config.calibration_samples,
        context.config.calibration_min_samples,
    )
    calibration_texts = load_text_samples(
        context.config.calibration_data,
        sample_count=requested_samples,
        split="train",
    )

    if len(calibration_texts) < context.config.calibration_min_samples:
        warning = (
            f"Calibration dataset has {len(calibration_texts)} samples; "
            f"recommended minimum is {context.config.calibration_min_samples}."
        )
        context.warnings.append(warning)
        stage_logger.warning(warning)

    return calibration_texts


def _log_quantization_plan(
    layers: list[tuple[str, torch.nn.Module]],
    plan: dict[str, int],
    stage_logger,
) -> None:
    """Log the planned layer-by-layer compression shape."""

    for layer_name, layer in layers:
        bits = plan[layer_name]
        before_size_gb = module_size_gb(layer)
        after_size_gb = _projected_layer_size_gb(layer, bits)
        stage_logger.info(
            "Layer %s processed: before_size_gb=%.6f after_size_gb=%.6f bits=%s",
            layer_name,
            before_size_gb,
            after_size_gb,
            bits,
        )


def _run_cpu_path(
    context: CompressionContext,
    stage_logger,
) -> dict[str, object]:
    """Execute the existing CPU quantization path without changing its logic."""

    layers = find_transformer_layers(context.model)
    layer_names = [layer_name for layer_name, _ in layers]
    calibration_texts = _load_calibration_texts(context, stage_logger)

    threshold = _plan_threshold(
        list(context.sensitivity_report.values()),
        context.config.sensitivity_quantile,
    )
    plan = build_quantization_plan(
        layer_names,
        context.sensitivity_report,
        threshold=threshold,
        default_bits=context.config.quant_default_bits,
        sensitive_bits=context.config.quant_sensitive_bits,
    )
    context.quantization_plan = plan
    _log_quantization_plan(layers, context.quantization_plan, stage_logger)
    context.update_progress("quantization", 0.2)

    model_source = str(getattr(context.model, "name_or_path", context.config.model_id))
    quantization_result = _run_cpu_backend(
        model_source=model_source,
        source_model=context.model,
        tokenizer=context.tokenizer,
        output_dir=context.config.output_dir,
        offload_dir=context.config.offload_dir,
        calibration_data=calibration_texts,
        calibration_source_name="wikitext2"
        if context.config.calibration_data is None
        else "custom",
        calibration_batch_size=max(1, context.config.calibration_batch_size),
        quantization_plan=context.quantization_plan,
        default_bits=context.config.quant_default_bits,
        gptq_group_size=context.config.gptq_group_size,
        gptq_desc_act=context.config.gptq_desc_act,
        gptq_damp_percent=context.config.gptq_damp_percent,
        trust_remote_code=context.config.trust_remote_code,
        hf_token=context.config.hf_token,
        stage_logger=stage_logger,
    )
    return {
        "result": quantization_result,
        "threshold": threshold,
        "calibration_samples": len(calibration_texts),
        "quantization_bits": {
            "default": context.config.quant_default_bits,
            "sensitive_layers": context.config.quant_sensitive_bits,
            "sensitive_threshold": GPTQ_SENSITIVE_THRESHOLD,
        },
        "sensitive_layers": [
            layer_name
            for layer_name, bits in context.quantization_plan.items()
            if bits == context.config.quant_sensitive_bits
        ],
    }


def _run_gptq_path(
    context: CompressionContext,
    stage_logger,
) -> dict[str, object]:
    """Execute the CUDA GPTQ quantization path."""

    layers = find_transformer_layers(context.model)
    layer_names = [layer_name for layer_name, _ in layers]
    calibration_texts = _load_calibration_texts(context, stage_logger)
    cpu_threshold = _plan_threshold(
        list(context.sensitivity_report.values()),
        context.config.sensitivity_quantile,
    )
    cpu_fallback_plan = build_quantization_plan(
        layer_names,
        context.sensitivity_report,
        threshold=cpu_threshold,
        default_bits=context.config.quant_default_bits,
        sensitive_bits=context.config.quant_sensitive_bits,
    )
    if context.sensitivity_report:
        context.quantization_plan = _build_bits_map(context.sensitivity_report)
    else:
        context.quantization_plan = {
            layer_name: 4 for layer_name, _ in layers
        }
    _log_quantization_plan(layers, context.quantization_plan, stage_logger)
    context.update_progress("quantization", 0.2)

    model_source = str(getattr(context.model, "name_or_path", context.config.model_id))
    quantization_result = _run_gptq_backend(
        model_source=model_source,
        source_model=context.model,
        tokenizer=context.tokenizer,
        output_dir=context.config.output_dir,
        offload_dir=context.config.offload_dir,
        calibration_data=calibration_texts,
        calibration_source_name="wikitext2"
        if context.config.calibration_data is None
        else "custom",
        calibration_batch_size=max(1, context.config.calibration_batch_size),
        quantization_plan=cpu_fallback_plan,
        default_bits=context.config.quant_default_bits,
        gptq_group_size=context.config.gptq_group_size,
        gptq_desc_act=context.config.gptq_desc_act,
        gptq_damp_percent=context.config.gptq_damp_percent,
        trust_remote_code=context.config.trust_remote_code,
        hf_token=context.config.hf_token,
        stage_logger=stage_logger,
        sensitivity_report=context.sensitivity_report,
    )
    if "quantization_plan" in quantization_result:
        context.quantization_plan = dict(quantization_result["quantization_plan"])

    return {
        "result": quantization_result,
        "threshold": GPTQ_SENSITIVE_THRESHOLD,
        "calibration_samples": len(calibration_texts),
        "quantization_bits": {
            "default": 4,
            "sensitive_layers": 8,
            "sensitive_threshold": GPTQ_SENSITIVE_THRESHOLD,
        },
        "sensitive_layers": [
            layer_name
            for layer_name, bits in context.quantization_plan.items()
            if bits == 8
        ],
    }


class MixedPrecisionQuantizerStage(PipelineStage):
    name = "quantization"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        if torch.cuda.is_available():
            execution = _run_gptq_path(context, stage_logger)
        else:
            execution = _run_cpu_path(context, stage_logger)

        quantization_result = execution["result"]

        context.model = quantization_result["model"]
        context.exported_model_path = (
            None
            if quantization_result["method"] == "gptq_cuda"
            else quantization_result.get("saved_path")
        )
        context.current_size_gb = quantization_result["final_size_gb"]
        context.warnings.extend(quantization_result.get("warnings", []))

        context.stage_details[self.name] = {
            "threshold": execution["threshold"],
            "quantization_method": quantization_result["method"],
            "target_size_gb": context.config.target_size_gb,
            "estimated_size_gb": quantization_result["final_size_gb"],
            "calibration_samples": execution["calibration_samples"],
            "calibration_batch_size": quantization_result["effective_batch_size"],
            "cuda_available": quantization_result.get("cuda_available", False),
            "gptq_config": {
                "bits": execution["quantization_bits"]["default"],
                "group_size": context.config.gptq_group_size,
                "desc_act": context.config.gptq_desc_act,
                "damp_percent": context.config.gptq_damp_percent,
            },
            "quantization_bits": execution["quantization_bits"],
            "sensitive_layers": execution["sensitive_layers"],
            "warnings": quantization_result.get("warnings", []),
        }

        if self.name not in context.stages_applied:
            context.stages_applied.append(self.name)

        write_json(
            context.config.output_dir / "quantization_plan.json",
            {
                "format": context.config.quantized_format,
                "layers": context.quantization_plan,
                "backend": quantization_result["method"],
                "estimated_size_gb": quantization_result["final_size_gb"],
                "gptq_config": context.stage_details[self.name]["gptq_config"],
                "quantization_bits": context.stage_details[self.name]["quantization_bits"],
                "cuda_available": context.stage_details[self.name]["cuda_available"],
                "warnings": quantization_result.get("warnings", []),
            },
        )
        stage_logger.info(
            "Quantization complete using %s.",
            quantization_result["method"],
        )
        context.update_progress(self.name, 1.0)
