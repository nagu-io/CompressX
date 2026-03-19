from __future__ import annotations

import math
import time
import logging
import uuid
from pathlib import Path
from typing import Any

import psutil
import torch

from compressx.checkpoints import CheckpointManager
from compressx.config import CompressionConfig
from compressx.context import CompressionContext, ProgressCallback
from compressx.exporter import export_model, write_compression_report
from compressx.logging_utils import configure_logging, get_stage_logger
from compressx.modeling import (
    estimate_model_size_gb,
    estimate_model_source_size_gb,
    load_model_and_tokenizer,
)
from compressx.runtime import detect_hardware, ensure_free_disk_space
from compressx.stages import (
    AccuracyEvaluatorStage,
    DistillationFineTunerStage,
    MixedPrecisionQuantizerStage,
    SensitivityAnalyzerStage,
    StructuralPrunerStage,
)
from compressx.stages.qep import QEPQuantizationStage
from compressx.utils.io import directory_size_gb, read_json
from compressx.utils.formatting import render_completion_summary
from compressx.utils.quantization import estimate_quantized_size_gb, match_quantization_bits

_REPORTABLE_STAGES = {"quantization", "qep", "pruning", "distillation"}
_STAGE_MEMORY_FACTORS = {
    "sensitivity": (0.35, 0.20),
    "quantization": (0.50, 0.25),
    "qep": (0.30, 0.10),
    "pruning": (0.20, 0.10),
    "distillation": (0.75, 0.30),
    "evaluation": (0.30, 0.15),
}


class CompressionPipeline:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.checkpoints = CheckpointManager(config.checkpoint_dir)

    def _estimate_qep_adjusted_size_gb(self, context: CompressionContext) -> float | None:
        qep_details = context.stage_details.get("qep", {})
        parameter_details = qep_details.get("qep_parameter_details")
        if not context.qep_applied or not isinstance(parameter_details, dict):
            return None

        state_dict = context.model.state_dict()
        total_bytes = 0.0
        for name, tensor in state_dict.items():
            if tensor.ndim < 2:
                total_bytes += tensor.numel() * 2
                continue
            bits = match_quantization_bits(name, context.quantization_plan)
            total_bytes += math.ceil(tensor.numel() * bits / 8)
            total_bytes += 32

        for detail in parameter_details.values():
            if not isinstance(detail, dict):
                continue
            parameter_name = detail.get("parameter_name")
            estimated_size_bytes = detail.get("estimated_size_bytes")
            if not isinstance(parameter_name, str) or estimated_size_bytes is None:
                continue
            tensor = state_dict.get(parameter_name)
            if tensor is None or tensor.ndim < 2:
                continue
            bits = match_quantization_bits(parameter_name, context.quantization_plan)
            total_bytes -= math.ceil(tensor.numel() * bits / 8) + 32
            total_bytes += float(estimated_size_bytes)

        return total_bytes / (1024**3)

    def _refresh_compression_metrics(self, context: CompressionContext) -> None:
        if context.quantization_plan:
            context.current_size_gb = estimate_quantized_size_gb(
                context.model.state_dict(),
                context.quantization_plan,
            )
        else:
            context.current_size_gb = estimate_model_size_gb(context.model)

        qep_estimated_size_gb = self._estimate_qep_adjusted_size_gb(context)
        if qep_estimated_size_gb is not None:
            context.stage_details.setdefault("qep", {})[
                "qep_estimated_size_gb"
            ] = qep_estimated_size_gb
            context.current_size_gb = min(
                float(context.current_size_gb or qep_estimated_size_gb),
                float(qep_estimated_size_gb),
            )

        if context.current_size_gb:
            context.compression_ratio = (context.original_size_gb or 0.0) / max(
                context.current_size_gb,
                1e-9,
            )

    def _build_optimization_profiles(self) -> list[dict[str, float | int | str]]:
        base = {
            "label": "baseline",
            "default_bits": int(self.config.quant_default_bits),
            "sensitive_bits": int(self.config.quant_sensitive_bits),
            "head_prune_threshold": float(self.config.head_prune_threshold),
            "redundancy_threshold": float(self.config.redundancy_threshold),
        }

        if self.config.aggressive:
            profiles = [
                {
                    "label": "aggressive",
                    "default_bits": 2,
                    "sensitive_bits": min(base["sensitive_bits"], 4),
                    "head_prune_threshold": max(base["head_prune_threshold"], 0.05),
                    "redundancy_threshold": min(base["redundancy_threshold"], 0.70),
                },
                {
                    "label": "aggressive-max",
                    "default_bits": 2,
                    "sensitive_bits": min(base["sensitive_bits"], 3),
                    "head_prune_threshold": max(base["head_prune_threshold"], 0.10),
                    "redundancy_threshold": min(base["redundancy_threshold"], 0.65),
                },
            ]
        else:
            profiles = [
                base,
                {
                    "label": "tight",
                    "default_bits": min(base["default_bits"], 3),
                    "sensitive_bits": min(base["sensitive_bits"], 4),
                    "head_prune_threshold": max(base["head_prune_threshold"], 0.05),
                    "redundancy_threshold": min(base["redundancy_threshold"], 0.80),
                },
                {
                    "label": "max",
                    "default_bits": 2,
                    "sensitive_bits": min(base["sensitive_bits"], 4),
                    "head_prune_threshold": max(base["head_prune_threshold"], 0.10),
                    "redundancy_threshold": min(base["redundancy_threshold"], 0.70),
                },
            ]

        unique_profiles: list[dict[str, float | int | str]] = []
        seen: set[tuple[int, int, float, float]] = set()
        for profile in profiles:
            key = (
                int(profile["default_bits"]),
                int(profile["sensitive_bits"]),
                float(profile["head_prune_threshold"]),
                float(profile["redundancy_threshold"]),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_profiles.append(profile)
        return unique_profiles

    def _apply_optimization_profile(
        self,
        profile: dict[str, float | int | str],
    ) -> None:
        self.config.quant_default_bits = int(profile["default_bits"])
        self.config.quant_sensitive_bits = int(profile["sensitive_bits"])
        self.config.head_prune_threshold = float(profile["head_prune_threshold"])
        self.config.redundancy_threshold = float(profile["redundancy_threshold"])

    def _run_target_optimization(self, context: CompressionContext) -> None:
        details = context.stage_details.setdefault(
            "target_optimization",
            {
                "target_size_gb": self.config.target_size_gb,
                "min_accuracy_retention_percent": self.config.min_accuracy_retention_percent,
                "passes": [],
            },
        )

        if self.config.skip_quantization and self.config.skip_pruning:
            context.target_stop_reason = "compression_stages_skipped"
            details["stop_reason"] = context.target_stop_reason
            return

        for index, profile in enumerate(self._build_optimization_profiles(), start=1):
            self._apply_optimization_profile(profile)

            if not self.config.skip_quantization:
                self._run_stage(
                    context,
                    "quantization",
                    MixedPrecisionQuantizerStage(),
                    save_model_checkpoint=False,
                )
                if context.use_qep:
                    self._run_stage(
                        context,
                        "qep",
                        QEPQuantizationStage(),
                        save_model_checkpoint=False,
                    )
            if not self.config.skip_pruning:
                self._run_stage(
                    context,
                    "pruning",
                    StructuralPrunerStage(),
                    save_model_checkpoint=True,
                )

            self._refresh_compression_metrics(context)

            if not self.config.skip_evaluation:
                self._run_stage(
                    context,
                    "evaluation",
                    AccuracyEvaluatorStage(),
                    save_model_checkpoint=False,
                )

            context.optimization_passes = index
            details["passes"].append(
                {
                    "pass_index": index,
                    "label": profile["label"],
                    "default_bits": self.config.quant_default_bits,
                    "sensitive_bits": self.config.quant_sensitive_bits,
                    "head_prune_threshold": self.config.head_prune_threshold,
                    "redundancy_threshold": self.config.redundancy_threshold,
                    "estimated_size_gb": context.current_size_gb,
                    "accuracy_retention_percent": context.accuracy_retention_percent,
                }
            )

            if (
                context.current_size_gb is not None
                and context.current_size_gb <= self.config.target_size_gb
            ):
                context.target_stop_reason = "size_target_met"
                break

            if (
                not self.config.skip_evaluation
                and context.accuracy_retention_percent is not None
                and context.accuracy_retention_percent
                < self.config.min_accuracy_retention_percent
            ):
                context.target_stop_reason = "accuracy_limit_reached"
                context.warnings.append(
                    "Target optimization stopped because estimated accuracy retention "
                    f"fell below {self.config.min_accuracy_retention_percent:.2f}%."
                )
                break
        else:
            context.target_stop_reason = "aggression_exhausted"

        details["stop_reason"] = context.target_stop_reason
        details["optimization_passes"] = context.optimization_passes

    def _hydrate_resume_artifacts(self, context: CompressionContext) -> None:
        sensitivity_path = context.config.output_dir / "sensitivity_report.json"
        head_path = context.config.output_dir / "head_sensitivity_report.json"
        quantization_path = context.config.output_dir / "quantization_plan.json"
        pruning_path = context.config.output_dir / "pruning_log.json"
        qep_path = context.config.output_dir / "qep_metadata.json"
        report_path = context.config.output_dir / "compression_report.json"

        if sensitivity_path.exists():
            context.sensitivity_report = {
                str(key): float(value)
                for key, value in read_json(sensitivity_path).items()
            }
        if head_path.exists():
            head_payload = read_json(head_path)
            context.head_report = {
                layer_name: {
                    int(head_index): float(score)
                    for head_index, score in head_scores.items()
                }
                for layer_name, head_scores in head_payload.items()
            }
        if quantization_path.exists():
            quant_payload = read_json(quantization_path)
            context.quantization_plan = {
                layer_name: int(bits)
                for layer_name, bits in quant_payload.get("layers", {}).items()
            }
        if pruning_path.exists():
            context.pruning_log = read_json(pruning_path)
        if qep_path.exists():
            qep_metadata = read_json(qep_path)
            context.qep_applied = bool(qep_metadata.get("layers_1bit", 0))
            context.qep_layers_1bit = int(qep_metadata.get("layers_1bit", 0))
            context.qep_layers_4bit = int(qep_metadata.get("layers_skipped", 0))
            context.qep_layers_8bit = 0
            context.stage_details["qep"] = {
                "qep_applied": context.qep_applied,
                "qep_layers_1bit": context.qep_layers_1bit,
                "qep_layers_4bit": context.qep_layers_4bit,
                "qep_layers_8bit": context.qep_layers_8bit,
                "qep_threshold": qep_metadata.get("threshold_used", context.qep_threshold),
                "qep_metadata_path": str(qep_path),
                "qep_estimated_weight_size_gb": float(
                    qep_metadata.get(
                        "estimated_weight_size_gb",
                        qep_metadata.get("estimated_size_gb", 0.0),
                    )
                ),
                "qep_layer_names": list(qep_metadata.get("layer_details", {}).keys()),
                "qep_parameter_details": {
                    layer_name: {
                        "parameter_name": detail.get("parameter_name"),
                        "estimated_size_bytes": float(
                            detail.get("estimated_size_bytes", 0.0)
                        ),
                    }
                    for layer_name, detail in qep_metadata.get("layer_details", {}).items()
                    if isinstance(detail, dict)
                },
            }
        if report_path.exists():
            report = read_json(report_path)
            context.current_size_gb = float(report.get("final_size_gb", 0.0))
            context.accuracy_retention_percent = float(
                report.get("accuracy_retention_percent", 0.0)
            )
            context.qep_applied = bool(report.get("qep_applied", context.qep_applied))
            context.qep_layers_1bit = int(report.get("qep_layers_1bit", context.qep_layers_1bit))
            context.qep_layers_4bit = int(report.get("qep_layers_4bit", context.qep_layers_4bit))
            context.qep_layers_8bit = int(report.get("qep_layers_8bit", context.qep_layers_8bit))

    def _check_disk_requirements(
        self,
        logger_adapter: logging.LoggerAdapter,
    ) -> None:
        estimated_size_gb = estimate_model_source_size_gb(
            self.config.model_id,
            hf_token=self.config.hf_token,
        )
        if estimated_size_gb is None:
            logger_adapter.warning(
                "Unable to estimate model size before loading; disk check is skipped."
            )
            return

        required_bytes = int(estimated_size_gb * 3 * (1024**3))
        ensure_free_disk_space(self.config.output_dir, required_bytes)
        logger_adapter.info(
            "Disk space check passed for %.2f GB model footprint estimate.",
            estimated_size_gb,
        )

    def _validate_preflight(self, logger_adapter: logging.LoggerAdapter) -> None:
        test_file = self.config.output_dir / f".write_test_{uuid.uuid4().hex}"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except OSError as exc:
            raise PermissionError(
                f"Output directory '{self.config.output_dir}' is not writable."
            ) from exc

        if self.config.calibration_data is not None:
            lines = [
                line.strip()
                for line in self.config.calibration_data.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            if len(lines) < self.config.calibration_min_samples:
                raise ValueError(
                    f"Calibration data must contain at least {self.config.calibration_min_samples} non-empty samples."
                )

        logger_adapter.info("Preflight validation passed for output and calibration inputs.")

    def _profile_memory(
        self,
        context: CompressionContext,
        stage_name: str,
    ) -> None:
        stage_logger = get_stage_logger(context.logger, stage_name)
        vm = psutil.virtual_memory()
        current_ram_used_gb = (vm.total - vm.available) / (1024**3)
        available_ram_gb = vm.available / (1024**3)
        ram_factor, vram_factor = _STAGE_MEMORY_FACTORS.get(stage_name, (0.25, 0.10))
        estimated_ram_needed_gb = max(
            0.1,
            ((context.runtime_size_gb or context.original_size_gb or 1.0) * ram_factor)
            + 0.1,
        )

        current_vram_used_gb = 0.0
        available_vram_gb = 0.0
        estimated_vram_needed_gb = 0.0
        if context.hardware_info.cuda_available and context.config.execution_device != "cpu":
            current_vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
            total_vram_gb = float(context.hardware_info.vram_gb or 0.0)
            available_vram_gb = max(total_vram_gb - current_vram_used_gb, 0.0)
            estimated_vram_needed_gb = max(
                0.1,
                ((context.runtime_size_gb or context.original_size_gb or 1.0) * vram_factor)
                + 0.1,
            )

        stage_logger.info(
            "Memory status: ram_used_gb=%.2f vram_used_gb=%.2f estimated_ram_needed_gb=%.2f",
            current_ram_used_gb,
            current_vram_used_gb,
            estimated_ram_needed_gb,
        )

        if available_ram_gb < estimated_ram_needed_gb:
            raise MemoryError(
                f"Insufficient RAM before stage '{stage_name}'. Available {available_ram_gb:.2f} GB, estimated need {estimated_ram_needed_gb:.2f} GB."
            )
        if estimated_vram_needed_gb > 0 and available_vram_gb < estimated_vram_needed_gb:
            raise MemoryError(
                f"Insufficient VRAM before stage '{stage_name}'. Available {available_vram_gb:.2f} GB, estimated need {estimated_vram_needed_gb:.2f} GB."
            )

    def _build_context(
        self,
        *,
        progress_callback: ProgressCallback | None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> CompressionContext:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.stage_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(self.config.log_file)
        general_logger = get_stage_logger(logger, "general")
        hardware = detect_hardware()
        general_logger.info(
            "Hardware detected: os=%s python=%s ram=%.2fGB device=%s gpu=%s vram=%s",
            hardware.os_name,
            hardware.python_version,
            hardware.ram_gb,
            hardware.execution_device,
            hardware.gpu_name,
            hardware.vram_gb,
        )
        if not hardware.cuda_available and self.config.execution_device == "auto":
            general_logger.warning("CUDA not available. Auto-switching to CPU mode.")
        self._validate_preflight(general_logger)
        self._check_disk_requirements(general_logger)

        resume_source: str | Path | None = None
        manifest = self.checkpoints.load_manifest() if self.config.resume else {}
        if self.config.resume:
            resume_source = self.checkpoints.latest_model_checkpoint()
            if resume_source is not None:
                general_logger.info(
                    "Resume requested. Loading from checkpoint at %s.",
                    resume_source,
                )

        if model is None or tokenizer is None:
            general_logger.info("Loading model %s", resume_source or self.config.model_id)
            model, tokenizer = load_model_and_tokenizer(
                self.config,
                hardware,
                model_source=resume_source,
            )

        context = CompressionContext(
            config=self.config,
            logger=logger,
            hardware_info=hardware,
            progress_callback=progress_callback,
            model=model,
            tokenizer=tokenizer,
            use_qep=self.config.use_qep,
            qep_threshold=self.config.qep_threshold,
        )
        context.runtime_size_gb = estimate_model_size_gb(model)
        context.original_size_gb = estimate_model_source_size_gb(
            resume_source or self.config.model_id,
            hf_token=self.config.hf_token,
        ) or context.runtime_size_gb
        context.stage_details["hardware"] = hardware.to_dict()
        context.stage_details["hardware"]["runtime_size_gb"] = context.runtime_size_gb
        if manifest:
            context.stages_applied = [
                stage_name
                for stage_name in manifest.get("completed_stages", [])
                if stage_name in _REPORTABLE_STAGES
            ]
            self._hydrate_resume_artifacts(context)
        return context

    def _run_stage(
        self,
        context: CompressionContext,
        stage_name: str,
        stage_instance,
        *,
        save_model_checkpoint: bool,
    ) -> None:
        if self.config.resume and self.checkpoints.is_stage_complete(stage_name):
            get_stage_logger(context.logger, stage_name).info(
                "Skipping stage because a completed checkpoint was found."
            )
            return

        self._profile_memory(context, stage_name)
        stage_started = time.perf_counter()
        stage_instance.run(context)
        duration = time.perf_counter() - stage_started
        details = context.stage_details.setdefault(stage_name, {})
        details["duration_seconds"] = duration
        self.checkpoints.record_stage(
            context,
            stage_name,
            save_model=save_model_checkpoint,
        )

    def run(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> CompressionContext:
        total_started = time.perf_counter()
        context = self._build_context(
            progress_callback=progress_callback,
            model=model,
            tokenizer=tokenizer,
        )

        if not self.config.skip_sensitivity:
            self._run_stage(
                context,
                "sensitivity",
                SensitivityAnalyzerStage(),
                save_model_checkpoint=False,
            )
        self._run_target_optimization(context)
        if self.config.distill:
            self._run_stage(
                context,
                "distillation",
                DistillationFineTunerStage(),
                save_model_checkpoint=True,
            )
            self._refresh_compression_metrics(context)

        export_model(context)
        self.checkpoints.record_stage(context, "export", save_model=False)

        if not self.config.skip_evaluation and self.config.distill:
            self._run_stage(
                context,
                "evaluation",
                AccuracyEvaluatorStage(),
                save_model_checkpoint=False,
            )

        actual_output_size_gb = directory_size_gb(context.config.output_dir)
        qep_estimated_size_gb = self._estimate_qep_adjusted_size_gb(context)
        if qep_estimated_size_gb is not None:
            context.stage_details.setdefault("qep", {})[
                "qep_estimated_size_gb"
            ] = qep_estimated_size_gb
            context.current_size_gb = min(
                float(actual_output_size_gb),
                float(qep_estimated_size_gb),
            )
        else:
            context.current_size_gb = actual_output_size_gb
        if context.current_size_gb:
            context.compression_ratio = (context.original_size_gb or 0.0) / max(
                context.current_size_gb,
                1e-9,
            )
        context.total_time_seconds = time.perf_counter() - total_started
        write_compression_report(context)
        get_stage_logger(context.logger, "general").info(
            "\n%s",
            render_completion_summary(context),
        )
        context.update_progress("complete", 1.0)
        return context
