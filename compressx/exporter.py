from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

from compressx.context import CompressionContext
from compressx.logging_utils import get_stage_logger
from compressx.reports import CompressionReport
from compressx.utils.io import directory_size_gb
from compressx.utils.quantization import quantize_state_dict


def export_model(context: CompressionContext) -> Path:
    stage_logger = get_stage_logger(context.logger, "export")
    config = context.config
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    quantization_method = context.stage_details.get("quantization", {}).get(
        "quantization_method"
    )

    if context.exported_model_path is not None and context.exported_model_path.exists():
        stage_logger.info(
            "Using model artifact exported during quantization stage at %s.",
            context.exported_model_path,
        )
        context.current_size_gb = directory_size_gb(output_dir)
        if context.current_size_gb:
            context.compression_ratio = (context.original_size_gb or 0.0) / max(
                context.current_size_gb,
                1e-9,
            )
        return context.exported_model_path

    model_path = output_dir / "model.safetensors"
    state_dict = context.model.state_dict()

    if quantization_method == "gptq_cuda" and hasattr(context.model, "save_pretrained"):
        context.model.save_pretrained(output_dir)
        if context.tokenizer is not None and hasattr(context.tokenizer, "save_pretrained"):
            context.tokenizer.save_pretrained(output_dir)
        context.exported_model_path = output_dir
        context.current_size_gb = directory_size_gb(output_dir)
        if context.current_size_gb:
            context.compression_ratio = (context.original_size_gb or 0.0) / max(
                context.current_size_gb,
                1e-9,
            )
        stage_logger.info("Exported GPTQ-quantized model to %s.", output_dir)
        return context.exported_model_path
    if context.quantization_plan and not config.skip_quantization:
        quantized_tensors = quantize_state_dict(state_dict, context.quantization_plan)
        save_file(quantized_tensors, str(model_path))
        stage_logger.info(
            "Exported quantized state dict to %s using packed SafeTensors.",
            model_path,
        )
    else:
        safe_state = {
            name: tensor.detach().cpu()
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
        }
        save_file(safe_state, str(model_path))
        stage_logger.info("Exported model weights to %s.", model_path)

    config_path = output_dir / "config.json"
    if hasattr(context.model, "config") and context.model.config is not None:
        context.model.config.to_json_file(config_path)
    else:
        config_path.write_text("{}", encoding="utf-8")

    if context.tokenizer is not None and hasattr(context.tokenizer, "save_pretrained"):
        context.tokenizer.save_pretrained(output_dir)

    context.exported_model_path = model_path
    context.current_size_gb = directory_size_gb(output_dir)
    if context.current_size_gb:
        context.compression_ratio = (context.original_size_gb or 0.0) / max(
            context.current_size_gb,
            1e-9,
        )
    return model_path


def write_compression_report(context: CompressionContext) -> Path:
    output_dir = context.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pruning_log = context.pruning_log or {}
    quantization_details = context.stage_details.get("quantization", {})
    qep_details = context.stage_details.get("qep", {})
    quantization_bits = quantization_details.get("quantization_bits")
    if not isinstance(quantization_bits, dict):
        quantization_bits = {
            "default": context.config.quant_default_bits
            if not context.config.skip_quantization
            else 16,
            "sensitive_layers": context.config.quant_sensitive_bits
            if not context.config.skip_quantization
            else 16,
            "sensitive_threshold": 0.7,
        }
    report = CompressionReport(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        model_id=context.config.model_id,
        model_architecture=(
            str(context.model.config.architectures[0])
            if hasattr(context.model, "config")
            and getattr(context.model.config, "architectures", None)
            else None
        ),
        original_size_gb=round(context.original_size_gb or 0.0, 3),
        final_size_gb=round(context.current_size_gb or 0.0, 3),
        compression_ratio=f"{round(context.compression_ratio or 0.0, 2)}x",
        accuracy_retention_percent=round(context.accuracy_retention_percent or 0.0, 2),
        perplexity_original=round(context.baseline_perplexity or 0.0, 4),
        perplexity_compressed=round(context.compressed_perplexity or 0.0, 4),
        inference_speed_original=f"{round(context.baseline_tokens_per_second or 0.0, 2)} tokens/sec",
        inference_speed_compressed=f"{round(context.compressed_tokens_per_second or 0.0, 2)} tokens/sec",
        total_time_minutes=round((context.total_time_seconds or 0.0) / 60.0, 2),
        stages_applied=context.stages_applied,
        layers_pruned=len(pruning_log.get("layers_removed", [])),
        heads_pruned=sum(
            len(heads) for heads in pruning_log.get("heads_removed", {}).values()
        ),
        quantization_method=quantization_details.get("quantization_method"),
        quantization_bits=quantization_bits,
        cuda_available=bool(
            quantization_details.get("cuda_available", context.hardware_info.cuda_available)
        ),
        qep_applied=context.qep_applied,
        qep_layers_1bit=context.qep_layers_1bit,
        qep_layers_4bit=context.qep_layers_4bit,
        qep_layers_8bit=context.qep_layers_8bit,
        qep_threshold=(
            float(qep_details.get("qep_threshold"))
            if qep_details.get("qep_threshold") is not None
            else (context.qep_threshold if context.qep_applied else None)
        ),
        target_size_gb=context.config.target_size_gb,
        target_outcome=context.target_stop_reason,
        optimization_passes=context.optimization_passes,
        warnings=context.warnings,
        hardware=context.hardware_info.to_dict(),
    )
    path = output_dir / "compression_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
