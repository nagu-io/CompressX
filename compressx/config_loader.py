from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from compressx.config import CompressionConfig


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("compress_config.yaml must contain a top-level mapping.")
    return payload


def _apply_nested_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    quantization = payload.get("quantization", {}) or {}
    pruning = payload.get("pruning", {}) or {}
    distillation = payload.get("distillation", {}) or {}

    normalized: dict[str, Any] = {
        "model_id": payload.get("model_id"),
        "output_dir": payload.get("output_dir", payload.get("output", "./compressed_model")),
        "target_size_gb": payload.get("target_size_gb"),
        "calibration_data": payload.get("calibration_data"),
        "domain_data": payload.get("domain_data"),
        "report": payload.get("report"),
        "resume": payload.get("resume"),
        "aggressive": payload.get("aggressive"),
        "use_qep": payload.get("use_qep"),
        "qep_threshold": payload.get("qep_threshold"),
        "quant_default_bits": quantization.get("default_bits"),
        "quant_sensitive_bits": quantization.get("sensitive_bits"),
        "head_prune_threshold": pruning.get("head_threshold"),
        "redundancy_threshold": pruning.get("layer_threshold"),
        "distill": distillation.get("enabled"),
        "distillation_steps": distillation.get("steps"),
        "min_accuracy_retention_percent": payload.get("min_accuracy_retention_percent"),
    }
    return {key: value for key, value in normalized.items() if value is not None}


def build_config(overrides: dict[str, Any], config_path: Path | None = None) -> CompressionConfig:
    yaml_config = _apply_nested_mapping(load_yaml_config(config_path))
    merged = {**yaml_config, **{key: value for key, value in overrides.items() if value is not None}}
    return CompressionConfig(**merged)
