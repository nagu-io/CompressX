from __future__ import annotations

import logging
import math
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_signs(weight: torch.Tensor) -> torch.Tensor:
    """
    Convert a weight tensor to a float sign tensor of +1.0 and -1.0 values.

    Zero weights map to +1.0 by convention.
    """

    return torch.where(weight < 0, -torch.ones_like(weight), torch.ones_like(weight)).to(
        torch.float32
    )


def compute_scale(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the per-output-channel mean absolute scale factor.

    For a 1D tensor this returns shape ``[1]``. For higher-rank tensors it returns
    one value per output channel.
    """

    float_weight = weight.float()
    if float_weight.dim() == 1:
        return float_weight.abs().mean().unsqueeze(0)
    reduction_dims = tuple(range(1, float_weight.dim()))
    return float_weight.abs().mean(dim=reduction_dims)


def quantize_weight_1bit(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor into sign values and per-channel scales.
    """

    signs = extract_signs(weight)
    scale = compute_scale(weight)
    return signs, scale


def reconstruct_weight(
    signs: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct an approximate weight tensor from signs and per-channel scales.
    """

    if signs.dim() == 1:
        return signs * scale
    shape = [-1] + [1] * (signs.dim() - 1)
    return signs * scale.view(shape)


def compute_quantization_error(
    original: torch.Tensor,
    signs: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the reconstruction error introduced by 1-bit quantization.
    """

    approximated = reconstruct_weight(signs, scale)
    return original - approximated


def _apply_compensation(weight: torch.Tensor, compensation: torch.Tensor) -> torch.Tensor:
    """
    Apply a compensation tensor to a weight matrix, broadcasting vectors when needed.
    """

    if compensation.shape == weight.shape:
        return weight + compensation
    if compensation.dim() == 1 and weight.dim() > 1 and compensation.shape[0] == weight.shape[0]:
        broadcast_shape = [-1] + [1] * (weight.dim() - 1)
        return weight + compensation.view(broadcast_shape)

    logger.warning(
        "Compensation shape mismatch %s vs %s, skipping",
        tuple(compensation.shape),
        tuple(weight.shape),
    )
    return weight


def estimate_qep_weight_size_bytes(
    signs: torch.Tensor,
    scale: torch.Tensor,
    compensation: torch.Tensor | None,
) -> int:
    """
    Estimate the conceptual packed size for a QEP-compressed weight tensor.

    Signs are treated as true 1-bit storage for size accounting.
    """

    sign_bytes = math.ceil(signs.numel() / 8)
    scale_bytes = scale.numel() * 2
    compensation_bytes = 0 if compensation is None else compensation.numel() * 2
    return sign_bytes + scale_bytes + compensation_bytes


class QEPQuantizer:
    """
    Sequential 1-bit quantizer with simple error-propagation compensation.
    """

    def __init__(self, sensitivity_threshold_1bit: float = 0.3):
        self.sensitivity_threshold_1bit = sensitivity_threshold_1bit
        self.compensation: Dict[int, torch.Tensor] = {}
        self.managed_parameters: Dict[str, str] = {}

    def should_apply_1bit(
        self,
        layer_name: str,
        sensitivity_scores: Dict[str, float],
    ) -> bool:
        """
        Return ``True`` when a layer is below the 1-bit sensitivity threshold.
        """

        score = sensitivity_scores.get(layer_name, 1.0)
        return score < self.sensitivity_threshold_1bit

    def quantize_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        sensitivity_scores: Dict[str, float],
        layer_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize a single layer weight tensor when it is safe for 1-bit QEP.
        """

        target_module, target_parameter_name = self._resolve_weight_target(layer)
        if target_module is None or not hasattr(target_module, "weight") or target_module.weight is None:
            return {}

        weight = target_module.weight.data.clone().float()

        if layer_idx in self.compensation:
            weight = _apply_compensation(weight, self.compensation[layer_idx])

        if not self.should_apply_1bit(layer_name, sensitivity_scores):
            logger.info(
                "Layer %s (%s): sensitivity too high, skipping 1-bit",
                layer_idx,
                layer_name,
            )
            return {}

        signs, scale = quantize_weight_1bit(weight)
        error = compute_quantization_error(weight, signs, scale)
        outgoing_compensation = (
            error.mean(dim=0)
            if error.dim() > 1
            else error.mean().unsqueeze(0)
        )
        self.compensation[layer_idx + 1] = outgoing_compensation

        logger.info(
            "Layer %s (%s): 1-bit applied. Scale range: [%.4f, %.4f]",
            layer_idx,
            layer_name,
            float(scale.min().item()),
            float(scale.max().item()),
        )
        self.managed_parameters[layer_name] = target_parameter_name

        return {
            "signs": signs.to(torch.int8),
            "scale": scale.to(torch.float16),
            "compensation": outgoing_compensation.to(torch.float16),
        }

    def quantize_model(
        self,
        model: nn.Module,
        sensitivity_scores: Dict[str, float],
        layers,
    ) -> tuple[Dict[str, Dict[str, torch.Tensor]], int, int]:
        """
        Apply QEP quantization to a sequence of layers in order.
        """

        del model
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        layers_1bit = 0
        layers_skipped = 0

        for idx, entry in enumerate(layers):
            if isinstance(entry, tuple) and len(entry) == 2:
                layer_name, layer = entry
            else:
                layer_name = f"layer_{idx}"
                layer = entry

            tensors = self.quantize_layer(
                idx,
                layer,
                sensitivity_scores,
                layer_name,
            )
            if tensors:
                results[layer_name] = tensors
                layers_1bit += 1
                signs = tensors["signs"].float()
                scale = tensors["scale"].float()
                target_module, _ = self._resolve_weight_target(layer)
                if target_module is not None and target_module.weight is not None:
                    target_module.weight.data = reconstruct_weight(signs, scale).to(
                        target_module.weight.dtype
                    )
            else:
                layers_skipped += 1

        logger.info(
            "QEP complete: %s layers at 1-bit, %s layers skipped",
            layers_1bit,
            layers_skipped,
        )
        return results, layers_1bit, layers_skipped

    def _resolve_weight_target(
        self,
        layer: nn.Module,
    ) -> tuple[nn.Module | None, str]:
        """
        Resolve the primary weight-bearing submodule for a transformer block.
        """

        if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor) and layer.weight.ndim >= 2:
            return layer, "weight"

        best_module: nn.Module | None = None
        best_parameter_name = ""
        best_numel = -1
        for module_name, module in layer.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
                continue
            if weight.numel() <= best_numel:
                continue
            best_module = module
            best_numel = weight.numel()
            best_parameter_name = f"{module_name}.weight" if module_name else "weight"

        return best_module, best_parameter_name
