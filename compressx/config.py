from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CompressionConfig:
    model_id: str
    output_dir: Path
    target_size_gb: float
    calibration_data: Path | None = None
    distill: bool = False
    domain_data: Path | None = None
    report: bool = False
    resume: bool = False
    aggressive: bool = False
    use_qep: bool = False
    qep_threshold: float = 0.3
    skip_sensitivity: bool = False
    skip_quantization: bool = False
    skip_pruning: bool = False
    skip_evaluation: bool = False
    sensitivity_dtype: str = "float16"
    quantized_format: str = "safetensors"
    calibration_samples: int = 128
    calibration_min_samples: int = 128
    calibration_batch_size: int = 4
    evaluation_samples: int = 16
    max_seq_len: int = 128
    head_prune_threshold: float = 0.01
    redundancy_threshold: float = 0.85
    sensitivity_quantile: float = 0.65
    sensitivity_passes: int = 32
    sensitivity_batch_size: int = 1
    quant_default_bits: int = 4
    quant_sensitive_bits: int = 8
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    gptq_damp_percent: float = 0.01
    gptq_use_triton: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    distillation_steps: int = 500
    learning_rate: float = 2e-4
    batch_size: int = 1
    temperature: float = 1.0
    min_accuracy_retention_percent: float = 90.0
    trust_remote_code: bool = False
    offload_dir: Path | None = None
    checkpoint_dir: Path | None = None
    log_file: Path = field(default_factory=lambda: Path("compress.log"))
    working_dir_name: str = ".compressx"
    checkpoint_shard_size: str = "2GB"
    hf_token: str | None = None
    execution_device: str = "auto"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.calibration_data:
            self.calibration_data = Path(self.calibration_data)
        if self.domain_data:
            self.domain_data = Path(self.domain_data)
        if self.offload_dir is None:
            self.offload_dir = Path("./offload")
        self.offload_dir = Path(self.offload_dir)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path("./checkpoints") / self.output_dir.name
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_file = Path(self.log_file)
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")

    @property
    def stage_dir(self) -> Path:
        return self.output_dir / self.working_dir_name
