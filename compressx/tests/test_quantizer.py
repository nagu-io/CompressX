from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from compressx.core.quantizer import (
    GPTQ_SENSITIVE_THRESHOLD,
    MixedPrecisionQuantizer,
    _build_bits_map,
    _run_gptq_path,
)
from compressx.exporter import write_compression_report
from compressx.stages.quantization import MixedPrecisionQuantizerStage
from compressx.utils.quantization import quantize_state_dict

from compressx.tests.conftest import TinyModel


def test_quantizer_happy_path(compression_context) -> None:
    compression_context.sensitivity_report = {
        "model.layers.0": 0.2,
        "model.layers.1": 0.9,
        "model.layers.2": 0.1,
    }
    stage = MixedPrecisionQuantizer()
    stage.run(compression_context)

    assert compression_context.quantization_plan["model.layers.1"] == 8
    assert (compression_context.config.output_dir / "quantization_plan.json").exists()


def test_quantizer_edge_case_tiny_model(
    compression_context,
    tokenizer,
) -> None:
    compression_context.model = TinyModel()
    compression_context.tokenizer = tokenizer
    compression_context.sensitivity_report = {"model.layers.0": 0.1}

    stage = MixedPrecisionQuantizer()
    stage.run(compression_context)

    assert compression_context.quantization_plan["model.layers.0"] == 4


def test_quantizer_failure_wrong_model_type(compression_context) -> None:
    compression_context.model = "invalid-model"
    stage = MixedPrecisionQuantizer()

    with pytest.raises(Exception):
        stage.run(compression_context)


def test_quantize_state_dict_supports_2_and_3_bit_exports(toy_model) -> None:
    quantized = quantize_state_dict(
        toy_model.state_dict(),
        {
            "model.layers.0": 2,
            "model.layers.1": 3,
            "model.layers.2": 4,
        },
    )

    assert any(name.endswith("__int2") for name in quantized)
    assert any(name.endswith("__int3") for name in quantized)
    assert any(name.endswith("__int4") for name in quantized)


def test_cuda_detection_routes_correctly(compression_context) -> None:
    stage = MixedPrecisionQuantizerStage()
    gpu_result = {
        "result": {
            "model": compression_context.model,
            "method": "gptq_cuda",
            "saved_path": compression_context.config.output_dir,
            "final_size_gb": 0.05,
            "effective_batch_size": 4,
            "warnings": [],
            "cuda_available": True,
        },
        "threshold": GPTQ_SENSITIVE_THRESHOLD,
        "calibration_samples": 4,
        "quantization_bits": {
            "default": 4,
            "sensitive_layers": 8,
            "sensitive_threshold": GPTQ_SENSITIVE_THRESHOLD,
        },
        "sensitive_layers": ["model.layers.1"],
    }
    cpu_result = {
        "result": {
            "model": compression_context.model,
            "method": "manual_cpu",
            "saved_path": None,
            "final_size_gb": 0.06,
            "effective_batch_size": 2,
            "warnings": [],
            "cuda_available": False,
        },
        "threshold": 0.5,
        "calibration_samples": 4,
        "quantization_bits": {
            "default": 4,
            "sensitive_layers": 8,
            "sensitive_threshold": GPTQ_SENSITIVE_THRESHOLD,
        },
        "sensitive_layers": ["model.layers.1"],
    }

    with (
        patch("compressx.stages.quantization.torch.cuda.is_available", return_value=True),
        patch("compressx.stages.quantization._run_gptq_path", return_value=gpu_result) as gpu_path,
        patch("compressx.stages.quantization._run_cpu_path", return_value=cpu_result) as cpu_path,
    ):
        stage.run(compression_context)
        gpu_path.assert_called_once()
        cpu_path.assert_not_called()

    compression_context.stage_details.clear()
    compression_context.stages_applied.clear()

    with (
        patch("compressx.stages.quantization.torch.cuda.is_available", return_value=False),
        patch("compressx.stages.quantization._run_gptq_path", return_value=gpu_result) as gpu_path,
        patch("compressx.stages.quantization._run_cpu_path", return_value=cpu_result) as cpu_path,
    ):
        stage.run(compression_context)
        cpu_path.assert_called_once()
        gpu_path.assert_not_called()


def test_gptq_import_error_fallback(compression_context) -> None:
    stage_logger = Mock()
    cpu_result = {
        "model": compression_context.model,
        "method": "manual_cpu",
        "saved_path": None,
        "final_size_gb": 0.06,
        "effective_batch_size": 4,
        "warnings": [],
        "quantization_plan": {},
        "cuda_available": False,
    }

    with (
        patch("compressx.core.quantizer._load_gptq_symbols", side_effect=ImportError("missing")),
        patch("compressx.core.quantizer._run_cpu_path", return_value=cpu_result) as cpu_path,
    ):
        result = _run_gptq_path(
            model_source="toy/model",
            source_model=compression_context.model,
            tokenizer=compression_context.tokenizer,
            output_dir=compression_context.config.output_dir,
            offload_dir=compression_context.config.offload_dir,
            calibration_data=["hello world"],
            calibration_source_name="wikitext2",
            calibration_batch_size=4,
            quantization_plan={"model.layers.0": 4},
            default_bits=4,
            gptq_group_size=128,
            gptq_desc_act=False,
            gptq_damp_percent=0.01,
            trust_remote_code=False,
            hf_token=None,
            stage_logger=stage_logger,
            sensitivity_report={"model.layers.0": 0.5},
        )

    assert result == cpu_result
    cpu_path.assert_called_once()
    assert any(
        "pip install transformers>=4.35.0 optimum auto-gptq" in str(call.args[0])
        for call in stage_logger.warning.call_args_list
    )


def test_sensitivity_bits_map() -> None:
    bits_map = _build_bits_map(
        {
            "layer_0": 0.85,
            "layer_1": 0.50,
            "layer_2": 0.15,
        }
    )

    assert bits_map == {
        "layer_0": 8,
        "layer_1": 4,
        "layer_2": 4,
    }


def test_oom_retry_reduces_batch_size(compression_context) -> None:
    stage_logger = Mock()
    attempted_batch_sizes: list[int] = []
    cpu_result = {
        "model": compression_context.model,
        "method": "manual_cpu",
        "saved_path": None,
        "final_size_gb": 0.06,
        "effective_batch_size": 4,
        "warnings": [],
        "quantization_plan": {},
        "cuda_available": False,
    }

    class FakeGPTQConfig:
        def __init__(self, **kwargs):
            self.batch_size = kwargs["batch_size"]

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_source: str, **kwargs):
            del model_source
            attempted_batch_sizes.append(kwargs["quantization_config"].batch_size)
            raise RuntimeError("CUDA out of memory")

    with (
        patch(
            "compressx.core.quantizer._load_gptq_symbols",
            return_value=(FakeGPTQConfig, FakeAutoModelForCausalLM),
        ),
        patch("compressx.core.quantizer.torch.cuda.is_available", return_value=True),
        patch("compressx.core.quantizer.torch.cuda.empty_cache"),
        patch("compressx.core.quantizer._run_cpu_path", return_value=cpu_result) as cpu_path,
    ):
        result = _run_gptq_path(
            model_source="toy/model",
            source_model=compression_context.model,
            tokenizer=compression_context.tokenizer,
            output_dir=compression_context.config.output_dir,
            offload_dir=compression_context.config.offload_dir,
            calibration_data=["hello world"],
            calibration_source_name="wikitext2",
            calibration_batch_size=4,
            quantization_plan={"model.layers.0": 4},
            default_bits=4,
            gptq_group_size=128,
            gptq_desc_act=False,
            gptq_damp_percent=0.01,
            trust_remote_code=False,
            hf_token=None,
            stage_logger=stage_logger,
            sensitivity_report={"model.layers.0": 0.5},
        )

    assert attempted_batch_sizes == [4, 2, 1]
    assert result == cpu_result
    cpu_path.assert_called_once()


def test_report_fields_populated(compression_context) -> None:
    compression_context.sensitivity_report = {
        "model.layers.0": 0.2,
        "model.layers.1": 0.9,
        "model.layers.2": 0.1,
    }
    stage = MixedPrecisionQuantizer()
    stage.run(compression_context)
    compression_context.original_size_gb = 1.0

    report_path = write_compression_report(compression_context)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["quantization_method"] == "manual_cpu"
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["quantization_bits"]["default"], int)
