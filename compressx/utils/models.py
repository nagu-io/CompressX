from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

ARCHITECTURE_MAP: dict[str, dict[str, str]] = {
    "LlamaForCausalLM": {
        "layers_path": "model.layers",
        "attention_attr": "self_attn",
        "num_heads_attr": "num_heads",
    },
    "MistralForCausalLM": {
        "layers_path": "model.layers",
        "attention_attr": "self_attn",
        "num_heads_attr": "num_heads",
    },
    "GPT2LMHeadModel": {
        "layers_path": "transformer.h",
        "attention_attr": "attn",
        "num_heads_attr": "num_heads",
    },
    "FalconForCausalLM": {
        "layers_path": "transformer.h",
        "attention_attr": "self_attention",
        "num_heads_attr": "num_heads",
    },
    "OPTForCausalLM": {
        "layers_path": "model.decoder.layers",
        "attention_attr": "self_attn",
        "num_heads_attr": "num_heads",
    },
    "BloomForCausalLM": {
        "layers_path": "transformer.h",
        "attention_attr": "self_attention",
        "num_heads_attr": "num_attention_heads",
    },
    "MPTForCausalLM": {
        "layers_path": "transformer.blocks",
        "attention_attr": "attn",
        "num_heads_attr": "n_heads",
    },
    "MixtralForCausalLM": {
        "layers_path": "model.layers",
        "attention_attr": "self_attn",
        "num_heads_attr": "num_heads",
    },
}

LAYER_CANDIDATE_PATHS = (
    "model.layers",
    "model.decoder.layers",
    "transformer.h",
    "gpt_neox.layers",
    "encoder.layer",
    "layers",
)


def get_module_by_path(model: torch.nn.Module, dotted_path: str) -> Any:
    """Resolve a dotted attribute path from a model instance."""

    current = model
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def get_arch_config(model: torch.nn.Module) -> dict[str, str]:
    """Return the architecture access configuration for a loaded model."""

    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) or []
    arch = str(architectures[0]) if architectures else "unknown"
    if arch not in ARCHITECTURE_MAP:
        supported = list(ARCHITECTURE_MAP.keys())
        raise ValueError(
            f"Architecture '{arch}' is not supported.\n"
            f"Supported architectures: {supported}\n"
            "Please open a GitHub issue to request support."
        )
    return ARCHITECTURE_MAP[arch]


def get_model_layers(model: torch.nn.Module) -> Any:
    """Resolve the architecture-specific layer collection for a model."""

    arch_config = get_arch_config(model)
    layers = get_module_by_path(model, arch_config["layers_path"])
    if layers is None:
        architectures = getattr(getattr(model, "config", None), "architectures", None) or []
        architecture_name = str(architectures[0]) if architectures else "unknown"
        raise ValueError(
            f"Unable to resolve layer path '{arch_config['layers_path']}' for architecture "
            f"'{architecture_name}'."
        )
    return layers


def get_attention_heads(
    layer: torch.nn.Module,
    arch_config: dict[str, str],
) -> torch.nn.Module:
    """Resolve the architecture-specific attention module for a layer."""

    return getattr(layer, arch_config["attention_attr"])


def get_num_heads(
    layer: torch.nn.Module,
    arch_config: dict[str, str],
) -> int:
    """Resolve the architecture-specific attention head count for a layer."""

    attention_module = get_attention_heads(layer, arch_config)
    return int(getattr(attention_module, arch_config["num_heads_attr"]))


def find_transformer_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Return named transformer layers, preferring architecture-aware resolution."""

    try:
        arch_config = get_arch_config(model)
        value = get_model_layers(model)
        if isinstance(value, Iterable):
            layers = list(value)
            if layers and all(isinstance(layer, torch.nn.Module) for layer in layers):
                return [
                    (f"{arch_config['layers_path']}.{index}", layer)
                    for index, layer in enumerate(layers)
                ]
    except ValueError:
        pass

    for dotted_path in LAYER_CANDIDATE_PATHS:
        value = get_module_by_path(model, dotted_path)
        if isinstance(value, torch.nn.ModuleList):
            return [(f"{dotted_path}.{index}", layer) for index, layer in enumerate(value)]
        if isinstance(value, Iterable):
            layers = list(value)
            if layers and all(isinstance(layer, torch.nn.Module) for layer in layers):
                return [(f"{dotted_path}.{index}", layer) for index, layer in enumerate(layers)]

    layers: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if any(token in name.lower() for token in ("layer", "block", "h.")):
            params = sum(parameter.numel() for parameter in module.parameters(recurse=False))
            if params:
                layers.append((name, module))
    return layers


def find_attention_module(
    layer: torch.nn.Module,
    arch_config: dict[str, str] | None = None,
) -> torch.nn.Module | None:
    """Find the attention module inside a transformer block."""

    if arch_config is not None:
        return getattr(layer, arch_config["attention_attr"], None)
    for name, module in layer.named_modules():
        lowered = name.lower()
        if "attn" in lowered or "attention" in lowered:
            return module
    return None


def find_layer_collection(
    model: torch.nn.Module,
) -> tuple[str | None, torch.nn.ModuleList | None]:
    """Resolve the mutable layer collection used for layer removal."""

    try:
        arch_config = get_arch_config(model)
        value = get_model_layers(model)
        if isinstance(value, torch.nn.ModuleList):
            return arch_config["layers_path"], value
    except ValueError:
        pass

    for dotted_path in LAYER_CANDIDATE_PATHS:
        value = get_module_by_path(model, dotted_path)
        if isinstance(value, torch.nn.ModuleList):
            return dotted_path, value
    return None, None


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def module_size_gb(module: torch.nn.Module) -> float:
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    return total_bytes / (1024**3)


def infer_vocab_size(model: torch.nn.Module) -> int:
    if hasattr(model, "config") and getattr(model.config, "vocab_size", None):
        return int(model.config.vocab_size)
    input_embeddings = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else None
    )
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return int(input_embeddings.weight.shape[0])
    if hasattr(model, "embed") and hasattr(model.embed, "weight"):
        return int(model.embed.weight.shape[0])
    return 32000
