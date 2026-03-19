from __future__ import annotations

import pytest

from compressx.core.analyzer import SensitivityAnalyzer

from compressx.tests.conftest import TinyModel


def test_analyzer_happy_path(compression_context) -> None:
    stage = SensitivityAnalyzer()
    stage.run(compression_context)

    assert compression_context.sensitivity_report
    assert (compression_context.config.output_dir / "sensitivity_report.json").exists()


def test_analyzer_edge_case_tiny_model(
    compression_context,
    tokenizer,
    hardware_info,
) -> None:
    compression_context.model = TinyModel()
    compression_context.tokenizer = tokenizer
    compression_context.hardware_info = hardware_info

    stage = SensitivityAnalyzer()
    stage.run(compression_context)

    assert len(compression_context.sensitivity_report) == 1


def test_analyzer_failure_wrong_model_type(compression_context) -> None:
    compression_context.model = "not-a-model"
    stage = SensitivityAnalyzer()

    with pytest.raises(Exception):
        stage.run(compression_context)
