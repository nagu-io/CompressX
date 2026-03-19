from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from tqdm.auto import tqdm

from compressx.context import CompressionContext
from compressx.logging_utils import get_stage_logger
from compressx.stages.base import PipelineStage
from compressx.utils.io import write_json
from compressx.utils.models import (
    get_arch_config,
    get_attention_heads,
    get_model_layers,
    get_num_heads,
    find_attention_module,
    find_transformer_layers,
    infer_vocab_size,
    module_size_gb,
)


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _primary_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _replace_primary_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        values = list(output)
        values[0] = new_tensor
        return tuple(values)
    return output


def _zero_module_output(
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
) -> Any:
    primary = _primary_tensor(output)
    if primary is None:
        return output
    return _replace_primary_tensor(output, torch.zeros_like(primary))


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {name: value / max_score for name, value in scores.items()}


def _quantile_threshold(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    clipped_fraction = min(max(fraction, 0.0), 1.0)
    index = min(
        len(ordered) - 1,
        max(0, int(math.floor(clipped_fraction * (len(ordered) - 1)))),
    )
    return ordered[index]


def _collect_attention_activations(
    layers: list[tuple[str, torch.nn.Module]],
    arch_config: dict[str, str] | None = None,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name, layer in layers:
        attention_module = (
            get_attention_heads(layer, arch_config)
            if arch_config is not None
            else find_attention_module(layer)
        )
        if attention_module is None:
            continue

        def capture_output(module, inputs, output, *, current_layer: str = layer_name):  # noqa: ANN001
            primary = _primary_tensor(output)
            if primary is not None:
                captured[current_layer] = primary.detach().cpu()

        hooks.append(attention_module.register_forward_hook(capture_output))
    return captured, hooks


def _compute_head_variances(
    attention_tensor: torch.Tensor,
    head_count: int,
) -> dict[int, float]:
    if head_count <= 0:
        return {}
    width = attention_tensor.shape[-1]
    head_dim = max(1, width // head_count)
    scores: dict[int, float] = {}
    for head_index in range(head_count):
        current_slice = attention_tensor[..., head_index * head_dim : (head_index + 1) * head_dim]
        if current_slice.numel() == 0:
            continue
        scores[head_index] = float(current_slice.var(unbiased=False).item())
    return scores


class SensitivityAnalyzerStage(PipelineStage):
    name = "sensitivity"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        model = context.model
        model.eval()

        arch_config: dict[str, str] | None = None
        try:
            arch_config = get_arch_config(model)
            layer_collection = get_model_layers(model)
            layers = [
                (f"{arch_config['layers_path']}.{index}", layer)
                for index, layer in enumerate(layer_collection)
            ]
        except ValueError:
            layers = find_transformer_layers(model)
        raw_layer_scores = {layer_name: 0.0 for layer_name, _ in layers}
        raw_head_scores: dict[str, dict[int, float]] = defaultdict(dict)
        device = _infer_input_device(model)
        vocab_size = infer_vocab_size(model)

        progress = tqdm(
            range(context.config.sensitivity_passes),
            desc="Stage 1 - Sensitivity",
            leave=False,
        )
        with torch.no_grad():
            for pass_index in progress:
                input_ids = torch.randint(
                    low=0,
                    high=max(vocab_size, 2),
                    size=(
                        context.config.sensitivity_batch_size,
                        context.config.max_seq_len,
                    ),
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)

                captured_activations, capture_hooks = _collect_attention_activations(
                    layers,
                    arch_config,
                )
                baseline_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                baseline_logits = baseline_outputs.logits.detach().cpu()
                for hook in capture_hooks:
                    hook.remove()

                for layer_name, layer in layers:
                    attention_module = (
                        get_attention_heads(layer, arch_config)
                        if arch_config is not None
                        else find_attention_module(layer)
                    )
                    if attention_module is not None and layer_name in captured_activations:
                        head_count = (
                            get_num_heads(layer, arch_config)
                            if arch_config is not None
                            else getattr(
                                attention_module,
                                "num_heads",
                                getattr(attention_module, "num_attention_heads", 0),
                            )
                        )
                        head_scores = _compute_head_variances(
                            captured_activations[layer_name],
                            head_count,
                        )
                        for head_index, score in head_scores.items():
                            raw_head_scores[layer_name][head_index] = (
                                raw_head_scores[layer_name].get(head_index, 0.0) + score
                            )

                    hook = layer.register_forward_hook(_zero_module_output)
                    try:
                        ablated_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                    finally:
                        hook.remove()

                    ablated_logits = ablated_outputs.logits.detach().cpu()
                    variance_score = float(
                        torch.var(
                            ablated_logits - baseline_logits,
                            unbiased=False,
                        ).item()
                    )
                    raw_layer_scores[layer_name] += variance_score

                context.update_progress(
                    self.name,
                    (pass_index + 1) / max(1, context.config.sensitivity_passes),
                )

        context.sensitivity_report = _normalize_scores(raw_layer_scores)
        context.head_report = {
            layer_name: _normalize_scores({str(head): score for head, score in head_scores.items()})
            for layer_name, head_scores in raw_head_scores.items()
        }
        context.head_report = {
            layer_name: {int(head): score for head, score in scores.items()}
            for layer_name, scores in context.head_report.items()
        }

        threshold = _quantile_threshold(
            list(context.sensitivity_report.values()),
            context.config.sensitivity_quantile,
        )
        stage_details = {
            "method": "random_input_zero_ablation_variance",
            "passes": context.config.sensitivity_passes,
            "threshold": threshold,
            "aggressive_quantization_layers": [
                name
                for name, score in context.sensitivity_report.items()
                if score < threshold
            ],
            "higher_precision_layers": [
                name
                for name, score in context.sensitivity_report.items()
                if score >= threshold
            ],
        }
        context.stage_details[self.name] = stage_details

        for layer_name, layer in layers:
            stage_logger.info(
                "Layer %s processed: before_size_gb=%.6f after_size_gb=%.6f sensitivity=%.6f",
                layer_name,
                module_size_gb(layer),
                module_size_gb(layer),
                context.sensitivity_report.get(layer_name, 0.0),
            )

        write_json(
            context.config.output_dir / "sensitivity_report.json",
            context.sensitivity_report,
        )
        write_json(
            context.config.output_dir / "head_sensitivity_report.json",
            {
                layer_name: {
                    str(head_index): score for head_index, score in scores.items()
                }
                for layer_name, scores in context.head_report.items()
            },
        )
        stage_logger.info(
            "Sensitivity analysis complete for %s layers.",
            len(context.sensitivity_report),
        )
        context.update_progress(self.name, 1.0)
