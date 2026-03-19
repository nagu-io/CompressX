from __future__ import annotations

from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from compressx.context import CompressionContext
from compressx.datasets import load_text_samples
from compressx.exceptions import ConfigurationError
from compressx.logging_utils import get_stage_logger
from compressx.modeling import load_teacher_model
from compressx.stages.base import PipelineStage
from compressx.utils.imports import optional_import


def _infer_lora_targets(model: torch.nn.Module) -> list[str]:
    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "dense",
        "up_proj",
        "down_proj",
        "gate_proj",
    }
    matches: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in preferred:
                matches.append(leaf)
    if matches:
        return sorted(set(matches))

    fallback: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fallback.append(name.split(".")[-1])
        if len(fallback) >= 4:
            break
    return sorted(set(fallback))


def _infer_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


class DistillationFineTunerStage(PipelineStage):
    name = "distillation"

    def run(self, context: CompressionContext) -> None:
        stage_logger = get_stage_logger(context.logger, self.name)
        if not context.config.distill:
            return
        if context.config.domain_data is None:
            raise ConfigurationError("--distill requires --domain-data.")

        peft = optional_import("peft")
        if peft is None:
            raise ConfigurationError(
                "peft must be installed to run the distillation stage."
            )

        teacher_model, _ = load_teacher_model(context.config, context.hardware_info)
        teacher_model.eval()
        teacher_device = _infer_device(teacher_model)
        student_device = _infer_device(context.model)

        target_modules = _infer_lora_targets(context.model)
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        student_model = peft.get_peft_model(context.model, lora_config)
        student_model.train()

        domain_texts = load_text_samples(
            context.config.domain_data,
            sample_count=max(context.config.distillation_steps, 8),
            split="train",
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student_model.parameters() if parameter.requires_grad),
            lr=context.config.learning_rate,
        )
        text_cycle = cycle(domain_texts)
        progress = tqdm(
            range(context.config.distillation_steps),
            desc="Stage 4 - Distillation",
            leave=False,
        )

        for step in progress:
            batch_texts = [next(text_cycle) for _ in range(context.config.batch_size)]
            batch = context.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=context.config.max_seq_len,
            )
            student_batch = {
                key: value.to(student_device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            teacher_batch = {
                key: value.to(teacher_device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }

            with torch.no_grad():
                teacher_logits = teacher_model(**teacher_batch).logits.detach()
            student_logits = student_model(**student_batch).logits

            seq_length = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :seq_length, :]
            teacher_logits = teacher_logits[:, :seq_length, :]

            temperature = context.config.temperature
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            context.update_progress(
                self.name,
                (step + 1) / context.config.distillation_steps,
            )

        adapter_dir = context.config.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(adapter_dir)
        merged_model = (
            student_model.merge_and_unload()
            if hasattr(student_model, "merge_and_unload")
            else student_model
        )

        context.model = merged_model
        context.adapter_path = Path(adapter_dir)
        context.stage_details[self.name] = {
            "steps": context.config.distillation_steps,
            "target_modules": target_modules,
            "adapter_path": str(adapter_dir),
        }
        if self.name not in context.stages_applied:
            context.stages_applied.append(self.name)
        stage_logger.info(
            "Distillation complete. Adapter saved to %s.",
            adapter_dir,
        )
        context.update_progress(self.name, 1.0)
