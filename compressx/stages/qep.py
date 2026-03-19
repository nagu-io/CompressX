from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compressx.context import CompressionContext
from compressx.core.qep_quantizer import (
    QEPQuantizer,
    estimate_qep_weight_size_bytes,
)
from compressx.logging_utils import get_stage_logger
from compressx.stages.base import PipelineStage
from compressx.utils.models import get_arch_config, get_model_layers


def run_qep_stage(
    model,
    output_dir: Path,
    sensitivity_report_path: Path,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Run the QEP 1-bit quantization pass on low-sensitivity layers.
    """

    if not sensitivity_report_path.exists():
        raise FileNotFoundError(
            f"Sensitivity report not found: {sensitivity_report_path}\n"
            "Run sensitivity analysis stage before QEP."
        )

    sensitivity_data = json.loads(sensitivity_report_path.read_text(encoding="utf-8"))
    sensitivity_scores = sensitivity_data.get("scores", sensitivity_data)
    if not isinstance(sensitivity_scores, dict):
        raise ValueError("Sensitivity report must contain a mapping of layer scores.")

    arch_config = get_arch_config(model)
    layers = list(get_model_layers(model))
    layers_path = arch_config.get("layers_path", "layer")
    named_layers = [
        (
            f"{layers_path}.{index}" if layers_path != "layer" else f"layer_{index}",
            layer,
        )
        for index, layer in enumerate(layers)
    ]
    normalized_scores = {
        str(key): float(value) for key, value in sensitivity_scores.items()
    }
    if named_layers and not any(
        normalized_scores.get(layer_name, 1.0) < threshold
        for layer_name, _ in named_layers
    ):
        lowest_name, _ = min(
            named_layers,
            key=lambda item: normalized_scores.get(item[0], 1.0),
        )
        normalized_scores[lowest_name] = min(
            normalized_scores.get(lowest_name, 1.0),
            max(threshold / 2.0, 0.0),
        )

    quantizer = QEPQuantizer(sensitivity_threshold_1bit=threshold)
    results, layers_1bit, layers_skipped = quantizer.quantize_model(
        model,
        normalized_scores,
        named_layers,
    )

    parameter_details: dict[str, dict[str, Any]] = {}
    for layer_name, tensors in results.items():
        estimated_size_bytes = estimate_qep_weight_size_bytes(
            tensors["signs"],
            tensors["scale"],
            tensors.get("compensation"),
        )
        parameter_details[layer_name] = {
            "parameter_name": (
                f"{layer_name}.{quantizer.managed_parameters[layer_name]}"
                if quantizer.managed_parameters.get(layer_name)
                else f"{layer_name}.weight"
            ),
            "estimated_size_bytes": estimated_size_bytes,
        }
    total_estimated_size_bytes = sum(
        detail["estimated_size_bytes"] for detail in parameter_details.values()
    )

    qep_meta = {
        "layers_1bit": layers_1bit,
        "layers_skipped": layers_skipped,
        "threshold_used": threshold,
        "estimated_size_bytes": total_estimated_size_bytes,
        "estimated_size_gb": total_estimated_size_bytes / (1024**3),
        "estimated_weight_size_gb": total_estimated_size_bytes / (1024**3),
        "layer_details": {
            layer_name: {
                "signs_shape": list(tensors["signs"].shape),
                "scale_shape": list(tensors["scale"].shape),
                "parameter_name": parameter_details[layer_name]["parameter_name"],
                "estimated_size_bytes": parameter_details[layer_name]["estimated_size_bytes"],
            }
            for layer_name, tensors in results.items()
        },
    }

    meta_path = output_dir / "qep_metadata.json"
    meta_path.write_text(json.dumps(qep_meta, indent=2), encoding="utf-8")

    return {
        "qep_applied": bool(results),
        "qep_layers_1bit": layers_1bit,
        "qep_layers_4bit": layers_skipped,
        "qep_layers_8bit": 0,
        "qep_threshold": threshold,
        "qep_metadata_path": str(meta_path),
        "qep_parameter_details": parameter_details,
        "qep_layer_names": list(results.keys()),
        "qep_estimated_weight_size_gb": total_estimated_size_bytes / (1024**3),
    }


class QEPQuantizationStage(PipelineStage):
    """Pipeline wrapper for the optional QEP quantization stage."""

    name = "qep"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        stage_logger.info("[STAGE] QEP 1-bit quantization")

        result = run_qep_stage(
            model=context.model,
            output_dir=context.config.output_dir,
            sensitivity_report_path=context.config.output_dir / "sensitivity_report.json",
            threshold=context.qep_threshold,
        )

        context.qep_applied = bool(result["qep_applied"])
        context.qep_layers_1bit = int(result["qep_layers_1bit"])
        context.qep_layers_4bit = int(result["qep_layers_4bit"])
        context.qep_layers_8bit = int(result["qep_layers_8bit"])
        context.stage_details[self.name] = result

        if self.name not in context.stages_applied:
            context.stages_applied.append(self.name)

        context.update_progress(self.name, 1.0)
