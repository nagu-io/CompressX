from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from compressx.exceptions import ConfigurationError
from compressx.utils.imports import optional_import


@dataclass(slots=True)
class HardwareInfo:
    os_name: str
    python_version: str
    ram_gb: float
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    execution_device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    psutil = optional_import("psutil")
    if psutil is not None:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    else:
        ram_gb = 0.0

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    vram_gb = None
    execution_device = "cpu"

    if cuda_available:
        device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        vram_gb = props.total_memory / (1024**3)
        execution_device = "cuda"

    return HardwareInfo(
        os_name=platform.system(),
        python_version=sys.version.split()[0],
        ram_gb=round(ram_gb, 2),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 2) if vram_gb is not None else None,
        execution_device=execution_device,
    )


def choose_execution_device(preference: str, hardware: HardwareInfo) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and not hardware.cuda_available:
        raise ConfigurationError("CUDA was requested but is not available.")
    if preference == "cuda":
        return "cuda"
    return "cuda" if hardware.cuda_available else "cpu"


def ensure_free_disk_space(target_dir: Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(target_dir.resolve().anchor or ".")
    if usage.free < required_bytes:
        required_gb = required_bytes / (1024**3)
        free_gb = usage.free / (1024**3)
        raise ConfigurationError(
            f"Insufficient disk space. Need {required_gb:.2f} GB free but only {free_gb:.2f} GB is available."
        )
