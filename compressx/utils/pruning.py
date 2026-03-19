from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from compressx.utils.models import (
    find_attention_module,
    find_layer_collection,
    find_transformer_layers,
    get_arch_config,
    get_attention_heads,
    get_num_heads,
)


def compute_redundancy_scores(model: torch.nn.Module) -> dict[str, float]:
    layers = find_transformer_layers(model)
    scores: dict[str, float] = {}
    previous_vector: torch.Tensor | None = None

    for layer_name, layer in layers:
        flattened_parameters = [
            parameter.detach().cpu().flatten() for parameter in layer.parameters()
        ]
        if not flattened_parameters:
            scores[layer_name] = 0.0
            continue
        current_vector = torch.cat(flattened_parameters)
        if previous_vector is None:
            scores[layer_name] = 0.0
        else:
            if previous_vector.numel() != current_vector.numel():
                width = min(previous_vector.numel(), current_vector.numel())
                previous_slice = previous_vector[:width]
                current_slice = current_vector[:width]
            else:
                previous_slice = previous_vector
                current_slice = current_vector
            scores[layer_name] = float(
                F.cosine_similarity(
                    previous_slice.unsqueeze(0),
                    current_slice.unsqueeze(0),
                ).item()
            )
        previous_vector = current_vector
    return scores


def _head_slices(head_count: int, width: int) -> list[slice]:
    head_dim = max(1, width // max(1, head_count))
    return [slice(index * head_dim, (index + 1) * head_dim) for index in range(head_count)]


def mask_attention_heads(
    attention_module: torch.nn.Module,
    heads: list[int],
    *,
    head_count: int | None = None,
) -> int:
    if not heads:
        return 0

    resolved_head_count = (
        head_count
        if head_count is not None
        else getattr(
            attention_module,
            "num_heads",
            getattr(attention_module, "num_attention_heads", 0),
        )
    )
    if resolved_head_count <= 0:
        return 0

    removed_params = 0
    projection_names = ("q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value")

    for projection_name in projection_names:
        projection = getattr(attention_module, projection_name, None)
        if not isinstance(projection, torch.nn.Linear):
            continue

        slices = _head_slices(resolved_head_count, projection.weight.shape[0])
        for head in heads:
            if head >= len(slices):
                continue
            current_slice = slices[head]
            projection.weight.data[current_slice, :] = 0
            removed_params += projection.weight.data[current_slice, :].numel()
            if projection.bias is not None:
                projection.bias.data[current_slice] = 0
                removed_params += projection.bias.data[current_slice].numel()
    return removed_params


def prune_model_heads(
    model: torch.nn.Module,
    head_scores: dict[str, dict[int, float]],
    threshold: float,
) -> tuple[dict[int, list[int]], int]:
    layers = find_transformer_layers(model)
    arch_config: dict[str, str] | None = None
    try:
        arch_config = get_arch_config(model)
    except ValueError:
        arch_config = None
    removal_map: dict[int, list[int]] = {}

    for layer_index, (layer_name, _) in enumerate(layers):
        candidates = [
            head_index
            for head_index, score in head_scores.get(layer_name, {}).items()
            if score < threshold
        ]
        if candidates:
            removal_map[layer_index] = sorted(set(candidates))

    if not removal_map:
        return {}, 0

    if hasattr(model, "prune_heads"):
        model.prune_heads(removal_map)
        estimated_params = sum(len(heads) for heads in removal_map.values())
        return removal_map, estimated_params

    removed_params = 0
    for layer_index, heads in removal_map.items():
        _, layer = layers[layer_index]
        attention_module = (
            get_attention_heads(layer, arch_config)
            if arch_config is not None
            else find_attention_module(layer)
        )
        if attention_module is None:
            continue
        head_count = (
            get_num_heads(layer, arch_config)
            if arch_config is not None
            else None
        )
        removed_params += mask_attention_heads(
            attention_module,
            heads,
            head_count=head_count,
        )
    return removal_map, removed_params


def remove_redundant_layers(
    model: torch.nn.Module,
    redundancy_scores: dict[str, float],
    threshold: float,
) -> tuple[list[int], int]:
    dotted_path, layers = find_layer_collection(model)
    if layers is None or dotted_path is None:
        return [], 0

    removable_indices = [
        index
        for index, (layer_name, _) in enumerate(find_transformer_layers(model))
        if index > 0 and redundancy_scores.get(layer_name, 0.0) > threshold
    ]
    if not removable_indices:
        return [], 0

    removed_params = 0
    for index in sorted(removable_indices, reverse=True):
        removed_layer = layers[index]
        removed_params += sum(
            parameter.numel() * parameter.element_size()
            for parameter in removed_layer.parameters()
        )
        del layers[index]

    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(layers)

    return removable_indices, removed_params


def estimate_size_savings_gb(parameter_or_byte_count: int, *, bytes_input: bool = False) -> float:
    total_bytes = parameter_or_byte_count if bytes_input else parameter_or_byte_count * 2
    return total_bytes / math.pow(1024, 3)
