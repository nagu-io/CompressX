from __future__ import annotations

from compressx.context import CompressionContext
from compressx.logging_utils import get_stage_logger
from compressx.stages.base import PipelineStage
from compressx.utils.io import write_json
from compressx.utils.models import find_transformer_layers
from compressx.utils.pruning import (
    compute_redundancy_scores,
    estimate_size_savings_gb,
    prune_model_heads,
    remove_redundant_layers,
)


class StructuralPrunerStage(PipelineStage):
    name = "pruning"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        layer_names = [
            layer_name for layer_name, _ in find_transformer_layers(context.model)
        ]
        context.redundancy_scores = compute_redundancy_scores(context.model)
        heads_removed, removed_head_params = prune_model_heads(
            context.model,
            context.head_report,
            context.config.head_prune_threshold,
        )
        layers_removed, removed_layer_bytes = remove_redundant_layers(
            context.model,
            context.redundancy_scores,
            context.config.redundancy_threshold,
        )

        estimated_savings = round(
            estimate_size_savings_gb(removed_head_params)
            + estimate_size_savings_gb(removed_layer_bytes, bytes_input=True),
            4,
        )
        context.pruning_log = {
            "heads_removed": {
                str(layer_index): heads for layer_index, heads in heads_removed.items()
            },
            "layers_removed": layers_removed,
            "redundancy_scores": context.redundancy_scores,
            "estimated_size_savings_gb": estimated_savings,
        }
        context.stage_details[self.name] = context.pruning_log

        for layer_index, heads in heads_removed.items():
            stage_logger.info(
                "Pruned attention heads from layer %s: removed=%s estimated_savings_gb=%.6f",
                layer_index,
                heads,
                estimated_savings,
            )
        for layer_index in layers_removed:
            layer_key = (
                layer_names[layer_index]
                if layer_index < len(layer_names)
                else str(layer_index)
            )
            stage_logger.info(
                "Removed transformer layer %s with redundancy score %.6f",
                layer_index,
                context.redundancy_scores.get(layer_key, 0.0),
            )

        if self.name not in context.stages_applied:
            context.stages_applied.append(self.name)

        write_json(context.config.output_dir / "pruning_log.json", context.pruning_log)
        stage_logger.info(
            "Pruning complete. Removed %s layers and %s head groups.",
            len(layers_removed),
            len(heads_removed),
        )
        context.update_progress(self.name, 1.0)
