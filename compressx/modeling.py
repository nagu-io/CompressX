from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch

from compressx.config import CompressionConfig
from compressx.exceptions import ConfigurationError
from compressx.runtime import HardwareInfo, choose_execution_device
from compressx.utils.imports import optional_import
from compressx.utils.io import directory_size_gb
from compressx.utils.models import ARCHITECTURE_MAP, get_arch_config

SUPPORTED_ARCHITECTURES = set(ARCHITECTURE_MAP.keys())


def resolve_dtype(dtype_name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return lookup[dtype_name.lower()]
    except KeyError as exc:
        raise ConfigurationError(f"Unsupported dtype: {dtype_name}") from exc


def infer_device_dtype(dtype_name: str, execution_device: str) -> torch.dtype:
    dtype = resolve_dtype(dtype_name)
    if execution_device == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def _resolve_model_class(
    transformers_module: Any,
    model_source: str,
    config: CompressionConfig,
) -> tuple[Any, Any, str]:
    source_path = Path(model_source)
    if source_path.exists():
        config_path = source_path / "config.json"
        if not config_path.exists():
            raise ConfigurationError(
                f"config.json was not found in local model directory '{source_path}'."
            )
        try:
            config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"config.json in '{source_path}' is not readable."
            ) from exc

    auto_config = transformers_module.AutoConfig.from_pretrained(
        model_source,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
    )
    architectures = getattr(auto_config, "architectures", None) or []
    architecture_name = next(
        (name for name in architectures if name in SUPPORTED_ARCHITECTURES),
        None,
    )
    if architecture_name is None:
        raise ConfigurationError(
            "Unsupported model architecture. CompressX currently supports "
            f"{', '.join(sorted(SUPPORTED_ARCHITECTURES))}."
        )
    model_class = (
        getattr(transformers_module, architecture_name)
        if architecture_name and hasattr(transformers_module, architecture_name)
        else transformers_module.AutoModelForCausalLM
    )
    return auto_config, model_class, architecture_name or "AutoModelForCausalLM"


def _build_max_memory(hardware: HardwareInfo, execution_device: str) -> dict[Any, str]:
    max_memory: dict[Any, str] = {}
    cpu_budget_gb = max(4, int(max(hardware.ram_gb * 0.75, 4)))
    max_memory["cpu"] = f"{cpu_budget_gb}GiB"
    if execution_device == "cuda" and hardware.vram_gb is not None:
        gpu_budget_gb = max(1, int(max(hardware.vram_gb * 0.85, 1)))
        max_memory[0] = f"{gpu_budget_gb}GiB"
    return max_memory


def _raise_model_access_error(model_source: str, exc: Exception) -> None:
    message = str(exc)
    if "gated" in message.lower() or "401" in message:
        raise ConfigurationError(
            f"Model '{model_source}' is gated or private. Set HF_TOKEN to access it."
        ) from exc
    if "404" in message or "not found" in message.lower():
        raise ConfigurationError(
            f"Model '{model_source}' was not found on Hugging Face."
        ) from exc
    raise ConfigurationError(f"Unable to load model '{model_source}': {message}") from exc


def load_model_and_tokenizer(
    config: CompressionConfig,
    hardware: HardwareInfo,
    *,
    model_source: str | Path | None = None,
) -> tuple[Any, Any]:
    logger = logging.getLogger("compressx")
    transformers_module = optional_import("transformers")
    accelerate = optional_import("accelerate")
    if transformers_module is None:
        raise ConfigurationError(
            "transformers must be installed to load Hugging Face models."
        )
    if accelerate is None:
        raise ConfigurationError(
            "accelerate must be installed to load large models with device_map='auto'."
        )

    resolved_source = str(model_source or config.model_id)
    execution_device = choose_execution_device(config.execution_device, hardware)
    config.offload_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            resolved_source,
            trust_remote_code=config.trust_remote_code,
            token=config.hf_token,
            use_fast=True,
        )
        auto_config, model_class, _ = _resolve_model_class(
            transformers_module,
            resolved_source,
            config,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "config": auto_config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": config.trust_remote_code,
            "device_map": "auto",
            "offload_folder": str(config.offload_dir),
            "offload_state_dict": True,
            "max_memory": _build_max_memory(hardware, execution_device),
            "torch_dtype": infer_device_dtype(config.sensitivity_dtype, execution_device),
            "token": config.hf_token,
        }
        try:
            model = model_class.from_pretrained(
                resolved_source,
                **model_kwargs,
            )
        except TypeError as exc:
            message = str(exc)
            if (
                "offload_weight() takes from 3 to 4 positional arguments but 5 were given"
                not in message
                or not model_kwargs.get("offload_state_dict")
            ):
                raise

            # transformers 4.38.0 can hit an incompatible CPU offload_state_dict path
            # on some environments. Retrying without that flag preserves device_map="auto"
            # and offload_folder support while avoiding the broken loader branch.
            model_kwargs["offload_state_dict"] = False
            model = model_class.from_pretrained(
                resolved_source,
                **model_kwargs,
            )
    except Exception as exc:
        _raise_model_access_error(resolved_source, exc)

    try:
        arch_config = get_arch_config(model)
        architecture_name = str(model.config.architectures[0])
        logger.info(
            "Architecture validated: %s (%s)",
            architecture_name,
            arch_config["layers_path"],
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    return model, tokenizer


def estimate_model_size_gb(model: torch.nn.Module) -> float:
    if hasattr(model, "get_memory_footprint"):
        try:
            return float(model.get_memory_footprint() / math.pow(1024, 3))
        except Exception:
            pass

    total_bytes = 0
    for parameter in model.state_dict().values():
        total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes / math.pow(1024, 3)


def estimate_model_source_size_gb(
    model_source: str | Path,
    *,
    hf_token: str | None = None,
) -> float | None:
    source_path = Path(model_source)
    if source_path.exists():
        return directory_size_gb(source_path)

    huggingface_hub = optional_import("huggingface_hub")
    if huggingface_hub is None:
        return None

    try:
        api = huggingface_hub.HfApi(token=hf_token)
        info = api.model_info(str(model_source), files_metadata=True, token=hf_token)
    except Exception:
        return None

    total_bytes = 0
    for sibling in getattr(info, "siblings", []):
        filename = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        if filename.endswith((".safetensors", ".bin", ".pt", ".gguf")):
            total_bytes += size

    if total_bytes == 0:
        return None
    return total_bytes / math.pow(1024, 3)


def load_teacher_model(
    config: CompressionConfig,
    hardware: HardwareInfo,
) -> tuple[Any, Any]:
    return load_model_and_tokenizer(config, hardware)


def estimate_export_size(output_dir: Path) -> float:
    return directory_size_gb(output_dir)
