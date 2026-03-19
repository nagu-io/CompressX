from __future__ import annotations

import math

import numpy as np
import torch


def _sanitize_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.item() == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return scale


def _validate_bits(bits: int) -> None:
    if bits not in {2, 3, 4, 8}:
        raise ValueError(f"Unsupported quantization bit-width: {bits}")


def _quantization_bounds(bits: int) -> tuple[int, int]:
    _validate_bits(bits)
    positive_max = (2 ** (bits - 1)) - 1
    negative_min = -(2 ** (bits - 1))
    return negative_min, positive_max


def quantize_int8_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_tensor = tensor.detach().cpu().to(torch.float32)
    scale = _sanitize_scale(cpu_tensor.abs().max() / 127.0)
    quantized = torch.clamp(torch.round(cpu_tensor / scale), -127, 127).to(torch.int8)
    return quantized, scale


def quantize_nbit_tensor(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_tensor = tensor.detach().cpu().to(torch.float32)
    negative_min, positive_max = _quantization_bounds(bits)
    scale = _sanitize_scale(cpu_tensor.abs().max() / max(positive_max, 1))
    quantized = torch.clamp(
        torch.round(cpu_tensor / scale),
        negative_min,
        positive_max,
    ).to(torch.int16)
    return quantized, scale


def _pack_nbit_values(values: torch.Tensor, bits: int) -> torch.Tensor:
    unsigned_values = (values.view(-1).to(torch.int32) + (2 ** (bits - 1))).cpu().numpy()
    bit_columns = ((unsigned_values[:, None] >> np.arange(bits)) & 1).astype(np.uint8)
    packed = np.packbits(bit_columns.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy())


def quantize_int4_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized, scale = quantize_nbit_tensor(tensor, 4)
    packed = _pack_nbit_values(quantized, 4)
    shape = torch.tensor(list(tensor.shape), dtype=torch.int32)
    return packed, scale, shape


def quantize_packed_tensor(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_bits(bits)
    if bits == 8:
        quantized, scale = quantize_int8_tensor(tensor)
        return quantized, scale, torch.tensor(list(tensor.shape), dtype=torch.int32)

    quantized, scale = quantize_nbit_tensor(tensor, bits)
    packed = _pack_nbit_values(quantized, bits)
    shape = torch.tensor(list(tensor.shape), dtype=torch.int32)
    return packed, scale, shape


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    quantized, scale = quantize_nbit_tensor(tensor, bits)
    restored = quantized.to(torch.float32) * scale
    return restored.to(dtype=tensor.dtype)


def apply_quantization_plan_inplace(
    model: torch.nn.Module,
    quantization_plan: dict[str, int],
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                continue
            if parameter.ndim < 2:
                continue
            bits = match_quantization_bits(name, quantization_plan)
            parameter.data.copy_(fake_quantize_tensor(parameter.data, bits))


def match_quantization_bits(param_name: str, quantization_plan: dict[str, int]) -> int:
    matched_bits = 8
    matched_length = -1
    for layer_name, bits in quantization_plan.items():
        if param_name.startswith(layer_name) and len(layer_name) > matched_length:
            matched_bits = bits
            matched_length = len(layer_name)
    return matched_bits


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    quantization_plan: dict[str, int],
) -> dict[str, torch.Tensor]:
    quantized_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            quantized_tensors[name] = tensor.detach().cpu()
            continue
        if tensor.ndim < 2:
            quantized_tensors[name] = tensor.detach().cpu().to(torch.float16)
            continue

        bits = match_quantization_bits(name, quantization_plan)
        if bits < 8:
            packed, scale, shape = quantize_packed_tensor(tensor, bits)
            quantized_tensors[f"{name}__int{bits}"] = packed
            quantized_tensors[f"{name}__scale"] = scale
            quantized_tensors[f"{name}__shape"] = shape
        else:
            quantized, scale = quantize_int8_tensor(tensor)
            quantized_tensors[f"{name}__int8"] = quantized
            quantized_tensors[f"{name}__scale"] = scale
            quantized_tensors[f"{name}__shape"] = torch.tensor(
                list(tensor.shape),
                dtype=torch.int32,
            )
    return quantized_tensors


def estimate_quantized_size_gb(
    state_dict: dict[str, torch.Tensor],
    quantization_plan: dict[str, int],
) -> float:
    total_bytes = 0
    for name, tensor in state_dict.items():
        if tensor.ndim < 2:
            total_bytes += tensor.numel() * 2
            continue
        bits = match_quantization_bits(name, quantization_plan)
        total_bytes += math.ceil(tensor.numel() * bits / 8)
        total_bytes += 32
    return total_bytes / math.pow(1024, 3)
