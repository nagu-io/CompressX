from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from compressx.core.qep_quantizer import (
    QEPQuantizer,
    compute_scale,
    extract_signs,
    quantize_weight_1bit,
    reconstruct_weight,
)
from compressx.exporter import write_compression_report
from compressx.stages.qep import run_qep_stage


def test_extract_signs_all_positive() -> None:
    weight = torch.tensor([0.5, 1.2, 0.0, 3.1])
    signs = extract_signs(weight)

    assert set(signs.tolist()) <= {1.0, -1.0}
    assert signs[2].item() == 1.0


def test_extract_signs_mixed() -> None:
    weight = torch.tensor([-0.3, 0.7, -1.5, 0.1])
    signs = extract_signs(weight)

    assert signs[0].item() == -1.0
    assert signs[1].item() == 1.0
    assert signs[2].item() == -1.0
    assert signs[3].item() == 1.0


def test_compute_scale_1d() -> None:
    weight = torch.tensor([2.0, -4.0, 1.0, -3.0])
    scale = compute_scale(weight)

    assert scale.shape == (1,)
    assert abs(scale.item() - 2.5) < 1e-4


def test_compute_scale_2d() -> None:
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    scale = compute_scale(weight)

    assert scale.shape == (2,)
    assert abs(scale[0].item() - 1.5) < 1e-4
    assert abs(scale[1].item() - 3.5) < 1e-4


def test_reconstruct_weight_roundtrip() -> None:
    torch.manual_seed(7)
    weight = torch.randn(8, 4)
    signs, scale = quantize_weight_1bit(weight)
    approximated = reconstruct_weight(signs, scale)

    assert approximated.shape == weight.shape
    mae = (weight - approximated).abs().mean()
    assert mae < weight.abs().std()


def test_error_propagation_reduces_error() -> None:
    torch.manual_seed(42)
    weight_1 = torch.randn(4, 4)
    weight_2 = torch.randn(4, 4)

    quantizer = QEPQuantizer(sensitivity_threshold_1bit=0.9)
    scores = {"layer_0": 0.1, "layer_1": 0.1}

    layer_1 = nn.Linear(4, 4, bias=False)
    layer_2 = nn.Linear(4, 4, bias=False)
    layer_1.weight.data = weight_1.clone()
    layer_2.weight.data = weight_2.clone()

    signs_2, scale_2 = quantize_weight_1bit(weight_2)
    error_without_qep = (weight_2 - reconstruct_weight(signs_2, scale_2)).abs().mean()

    results, layers_1bit, layers_skipped = quantizer.quantize_model(
        nn.Module(),
        scores,
        [layer_1, layer_2],
    )

    assert error_without_qep.item() >= 0.0
    assert layers_1bit == 2
    assert layers_skipped == 0
    assert "layer_1" in results
    assert 1 in quantizer.compensation
    assert quantizer.compensation[1].shape[0] == weight_1.shape[0]


def test_high_sensitivity_layer_skipped() -> None:
    quantizer = QEPQuantizer(sensitivity_threshold_1bit=0.3)
    scores = {"layer_0": 0.85}

    assert quantizer.should_apply_1bit("layer_0", scores) is False


def test_low_sensitivity_layer_applied() -> None:
    quantizer = QEPQuantizer(sensitivity_threshold_1bit=0.3)
    scores = {"layer_0": 0.15}

    assert quantizer.should_apply_1bit("layer_0", scores) is True


def test_missing_sensitivity_score_defaults_safe() -> None:
    quantizer = QEPQuantizer(sensitivity_threshold_1bit=0.3)

    assert quantizer.should_apply_1bit("layer_unknown", {}) is False


def test_qep_report_fields_present(tmp_path: Path) -> None:
    sensitivity_path = tmp_path / "sensitivity_report.json"
    sensitivity_path.write_text(
        json.dumps({"layer_0": 0.1, "layer_1": 0.8}),
        encoding="utf-8",
    )

    layer = nn.Linear(4, 4, bias=False)
    mock_model = MagicMock()

    with (
        patch("compressx.stages.qep.get_model_layers", return_value=[layer]),
        patch("compressx.stages.qep.get_arch_config", return_value={}),
    ):
        result = run_qep_stage(
            model=mock_model,
            output_dir=tmp_path,
            sensitivity_report_path=sensitivity_path,
            threshold=0.3,
        )

    metadata_path = tmp_path / "qep_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result["qep_applied"] is True
    assert result["qep_layers_1bit"] == 1
    assert "qep_layers_4bit" in result
    assert metadata_path.exists()
    assert "layer_details" in metadata
    assert "layer_0" in metadata["layer_details"]
    assert metadata["layer_details"]["layer_0"]["estimated_size_bytes"] > 0


def test_qep_report_fields_written(compression_context) -> None:
    compression_context.original_size_gb = 1.0
    compression_context.current_size_gb = 0.4
    compression_context.compression_ratio = 2.5
    compression_context.qep_applied = True
    compression_context.qep_layers_1bit = 2
    compression_context.qep_layers_4bit = 1
    compression_context.qep_layers_8bit = 0
    compression_context.stage_details["qep"] = {"qep_threshold": 0.3}

    report_path = write_compression_report(compression_context)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["qep_applied"] is True
    assert report["qep_layers_1bit"] == 2
    assert report["qep_layers_4bit"] == 1
    assert report["qep_layers_8bit"] == 0
    assert report["qep_threshold"] == 0.3
