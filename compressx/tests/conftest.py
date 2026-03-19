from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressx.config import CompressionConfig
from compressx.context import CompressionContext
from compressx.logging_utils import configure_logging
from compressx.runtime import HardwareInfo


class ToyConfig:
    def __init__(self, num_hidden_layers: int, hidden_size: int = 16, vocab_size: int = 64):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "num_hidden_layers": self.num_hidden_layers,
                    "hidden_size": self.hidden_size,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        rows = []
        for text in texts:
            token_ids = [((ord(char) % 31) + 1) for char in text]
            token_ids = token_ids[:max_length] if truncation else token_ids
            if not token_ids:
                token_ids = [1]
            rows.append(torch.tensor(token_ids, dtype=torch.long))

        width = max(row.shape[0] for row in rows)
        padded_rows = []
        for row in rows:
            pad_width = width - row.shape[0]
            padded_rows.append(F.pad(row, (0, pad_width), value=self.pad_token_id))

        input_ids = torch.stack(padded_rows)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")


class ToyAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return self.o_proj(attended)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.self_attn = ToyAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states + self.self_attn(hidden_states)
        return self.norm(residual + self.ff(residual))


class ToyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [ToyBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = ToyConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self.pruned_heads: dict[int, list[int]] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_logits.numel() == 0:
                loss = logits.mean() * 0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        return SimpleNamespace(logits=logits, loss=loss)

    def prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        self.pruned_heads = heads_to_prune


class TinyModel(ToyCausalLM):
    def __init__(self):
        super().__init__(num_layers=1, num_heads=1, hidden_size=8, vocab_size=64)


@pytest.fixture()
def tokenizer() -> ToyTokenizer:
    return ToyTokenizer()


@pytest.fixture()
def toy_model() -> ToyCausalLM:
    model = ToyCausalLM()
    with torch.no_grad():
        source = model.model.layers[0].state_dict()
        model.model.layers[1].load_state_dict(source)
    return model


@pytest.fixture()
def compression_config(tmp_path: Path) -> CompressionConfig:
    return CompressionConfig(
        model_id="toy/model",
        output_dir=tmp_path / "compressed_model",
        target_size_gb=3.0,
        calibration_samples=4,
        calibration_min_samples=4,
        calibration_batch_size=2,
        evaluation_samples=4,
        sensitivity_passes=4,
        max_seq_len=32,
        log_file=tmp_path / "compress.log",
    )


@pytest.fixture()
def hardware_info() -> HardwareInfo:
    return HardwareInfo(
        os_name="Windows",
        python_version="3.11.9",
        ram_gb=16.0,
        cuda_available=False,
        gpu_name=None,
        vram_gb=None,
        execution_device="cpu",
    )


@pytest.fixture()
def compression_context(
    compression_config: CompressionConfig,
    toy_model: ToyCausalLM,
    tokenizer: ToyTokenizer,
    hardware_info: HardwareInfo,
) -> CompressionContext:
    logger = configure_logging(compression_config.log_file)
    return CompressionContext(
        config=compression_config,
        logger=logger,
        hardware_info=hardware_info,
        model=toy_model,
        tokenizer=tokenizer,
    )
