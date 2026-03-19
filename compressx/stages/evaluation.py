from __future__ import annotations

import math
import time

import torch
from tqdm.auto import tqdm

from compressx.context import CompressionContext
from compressx.datasets import load_text_samples
from compressx.logging_utils import get_stage_logger
from compressx.modeling import load_teacher_model
from compressx.stages.base import PipelineStage


def _infer_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    *,
    max_seq_len: int,
    description: str,
) -> tuple[float, float]:
    device = _infer_device(model)
    model.eval()
    losses: list[float] = []
    total_tokens = 0
    started = time.perf_counter()

    progress = tqdm(texts, desc=description, leave=False)
    with torch.no_grad():
        for text in progress:
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            batch["labels"] = batch["input_ids"].clone()
            outputs = model(**batch)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
            total_tokens += int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else int(batch["input_ids"].numel())

    mean_loss = sum(losses) / max(1, len(losses))
    elapsed = max(time.perf_counter() - started, 1e-6)
    tokens_per_second = total_tokens / elapsed
    return float(math.exp(mean_loss)), float(tokens_per_second)


class AccuracyEvaluatorStage(PipelineStage):
    name = "evaluation"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        texts = load_text_samples(
            None,
            sample_count=context.config.evaluation_samples,
            split="train",
        )
        teacher_model, _ = load_teacher_model(context.config, context.hardware_info)

        baseline, baseline_speed = compute_perplexity(
            teacher_model,
            context.tokenizer,
            texts,
            max_seq_len=context.config.max_seq_len,
            description="Stage 5 - Baseline",
        )
        compressed, compressed_speed = compute_perplexity(
            context.model,
            context.tokenizer,
            texts,
            max_seq_len=context.config.max_seq_len,
            description="Stage 5 - Compressed",
        )
        retention = min(100.0, (baseline / max(compressed, 1e-6)) * 100.0)

        context.baseline_perplexity = baseline
        context.compressed_perplexity = compressed
        context.baseline_tokens_per_second = baseline_speed
        context.compressed_tokens_per_second = compressed_speed
        context.accuracy_retention_percent = retention
        context.stage_details[self.name] = {
            "benchmark": "wikitext-2-raw-v1",
            "baseline_perplexity": baseline,
            "compressed_perplexity": compressed,
            "accuracy_retention_percent": retention,
            "baseline_tokens_per_second": baseline_speed,
            "compressed_tokens_per_second": compressed_speed,
        }
        stage_logger.info(
            "Evaluation complete. Accuracy retention: %.2f%%, baseline_speed=%.2f tok/s, compressed_speed=%.2f tok/s.",
            retention,
            baseline_speed,
            compressed_speed,
        )
        context.update_progress(self.name, 1.0)
