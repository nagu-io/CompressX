from __future__ import annotations

import pytest

from compressx.core.pruner import StructuralPruner

from compressx.tests.conftest import TinyModel


def test_pruner_happy_path(compression_context) -> None:
    compression_context.head_report = {
        "model.layers.0": {0: 0.001, 1: 0.9},
        "model.layers.1": {0: 0.5},
        "model.layers.2": {0: 0.5},
    }
    compression_context.config.head_prune_threshold = 0.01
    compression_context.config.redundancy_threshold = 0.8

    stage = StructuralPruner()
    stage.run(compression_context)

    assert compression_context.pruning_log["heads_removed"]["0"] == [0]


def test_pruner_edge_case_tiny_model(
    compression_context,
    tokenizer,
) -> None:
    compression_context.model = TinyModel()
    compression_context.tokenizer = tokenizer
    compression_context.head_report = {"model.layers.0": {0: 1.0}}

    stage = StructuralPruner()
    stage.run(compression_context)

    assert compression_context.pruning_log["layers_removed"] == []


def test_pruner_failure_wrong_model_type(compression_context) -> None:
    compression_context.model = "bad-model"
    stage = StructuralPruner()

    with pytest.raises(Exception):
        stage.run(compression_context)
