from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from compressx.config import CompressionConfig
from compressx.runtime import HardwareInfo

ProgressCallback = Callable[[str, float, "CompressionContext"], None]


@dataclass
class CompressionContext:
    config: CompressionConfig
    logger: logging.Logger
    hardware_info: HardwareInfo
    progress_callback: ProgressCallback | None = None
    model: Any = None
    tokenizer: Any = None
    use_qep: bool = False
    qep_threshold: float = 0.3
    qep_applied: bool = False
    qep_layers_1bit: int = 0
    qep_layers_4bit: int = 0
    qep_layers_8bit: int = 0
    sensitivity_report: dict[str, float] = field(default_factory=dict)
    head_report: dict[str, dict[int, float]] = field(default_factory=dict)
    quantization_plan: dict[str, int] = field(default_factory=dict)
    pruning_log: dict[str, Any] = field(default_factory=dict)
    redundancy_scores: dict[str, float] = field(default_factory=dict)
    baseline_perplexity: float | None = None
    compressed_perplexity: float | None = None
    baseline_tokens_per_second: float | None = None
    compressed_tokens_per_second: float | None = None
    accuracy_retention_percent: float | None = None
    original_size_gb: float | None = None
    runtime_size_gb: float | None = None
    current_size_gb: float | None = None
    compression_ratio: float | None = None
    stages_applied: list[str] = field(default_factory=list)
    exported_model_path: Path | None = None
    adapter_path: Path | None = None
    stage_details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    total_time_seconds: float | None = None
    optimization_passes: int = 0
    target_stop_reason: str | None = None

    def update_progress(self, stage: str, progress: float) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, progress, self)
