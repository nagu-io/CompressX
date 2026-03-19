from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from compressx.exporter import write_compression_report
from compressx.utils.models import (
    ARCHITECTURE_MAP,
    get_arch_config,
    get_model_layers,
    get_num_heads,
)


def test_all_architectures_in_map() -> None:
    expected_architectures = {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GPT2LMHeadModel",
        "FalconForCausalLM",
        "OPTForCausalLM",
        "BloomForCausalLM",
        "MPTForCausalLM",
        "MixtralForCausalLM",
    }

    assert set(ARCHITECTURE_MAP.keys()) == expected_architectures
    for config in ARCHITECTURE_MAP.values():
        assert {"layers_path", "attention_attr", "num_heads_attr"} <= set(config.keys())


def test_get_arch_config_known_arch() -> None:
    model = MagicMock()
    model.config.architectures = ["LlamaForCausalLM"]

    arch_config = get_arch_config(model)

    assert arch_config == ARCHITECTURE_MAP["LlamaForCausalLM"]


def test_get_arch_config_unknown_arch() -> None:
    model = MagicMock()
    model.config.architectures = ["UnknownModelForCausalLM"]

    with pytest.raises(ValueError, match="not supported") as exc_info:
        get_arch_config(model)

    message = str(exc_info.value)
    assert "Supported architectures" in message
    assert "LlamaForCausalLM" in message
    assert "MixtralForCausalLM" in message


def test_get_model_layers_dotted_path() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["OPTForCausalLM"]),
        model=SimpleNamespace(
            decoder=SimpleNamespace(layers=[1, 2, 3]),
        ),
    )

    assert get_model_layers(model) == [1, 2, 3]


def test_get_model_layers_shallow_path() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        model=SimpleNamespace(layers=[4, 5, 6]),
    )

    assert get_model_layers(model) == [4, 5, 6]


def test_get_num_heads_resolves_attr() -> None:
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(num_heads=32),
    )
    arch_config = {
        "attention_attr": "self_attn",
        "num_heads_attr": "num_heads",
    }

    assert get_num_heads(layer, arch_config) == 32


def test_architecture_in_report(compression_context) -> None:
    compression_context.model.config.architectures = ["LlamaForCausalLM"]
    compression_context.original_size_gb = 1.0
    compression_context.current_size_gb = 0.5
    compression_context.compression_ratio = 2.0

    report_path = write_compression_report(compression_context)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["model_architecture"] == "LlamaForCausalLM"
