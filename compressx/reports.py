from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class CompressionReport:
    timestamp: str
    model_id: str
    model_architecture: str | None
    original_size_gb: float
    final_size_gb: float
    compression_ratio: str
    accuracy_retention_percent: float
    perplexity_original: float
    perplexity_compressed: float
    inference_speed_original: str
    inference_speed_compressed: str
    total_time_minutes: float
    stages_applied: list[str]
    layers_pruned: int
    heads_pruned: int
    quantization_method: str | None
    quantization_bits: dict[str, int | float]
    cuda_available: bool
    qep_applied: bool
    qep_layers_1bit: int
    qep_layers_4bit: int
    qep_layers_8bit: int
    qep_threshold: float | None
    target_size_gb: float
    target_outcome: str | None
    optimization_passes: int = 0
    warnings: list[str] = field(default_factory=list)
    hardware: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
