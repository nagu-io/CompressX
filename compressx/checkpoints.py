from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from compressx.context import CompressionContext
from compressx.utils.io import read_json, write_json


class CheckpointManager:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root_dir / "manifest.json"

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"completed_stages": [], "latest_model_checkpoint": None}
        return read_json(self.manifest_path)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.manifest_path, manifest)

    def is_stage_complete(self, stage_name: str) -> bool:
        manifest = self.load_manifest()
        return stage_name in manifest.get("completed_stages", [])

    def latest_model_checkpoint(self) -> Path | None:
        manifest = self.load_manifest()
        checkpoint = manifest.get("latest_model_checkpoint")
        return Path(checkpoint) if checkpoint else None

    def record_stage(
        self,
        context: CompressionContext,
        stage_name: str,
        *,
        save_model: bool,
    ) -> None:
        stage_dir = self.root_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "stage": stage_name,
            "current_size_gb": context.current_size_gb,
            "original_size_gb": context.original_size_gb,
            "stages_applied": context.stages_applied,
            "stage_details": context.stage_details.get(stage_name, {}),
        }
        write_json(stage_dir / "metadata.json", metadata)

        model_checkpoint_path: str | None = None
        if save_model:
            model_dir = stage_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_checkpoint_path = str(self._save_model(context, model_dir))

        manifest = self.load_manifest()
        completed = manifest.setdefault("completed_stages", [])
        if stage_name not in completed:
            completed.append(stage_name)
        if model_checkpoint_path is not None:
            manifest["latest_model_checkpoint"] = model_checkpoint_path
        self.save_manifest(manifest)

    def _save_model(self, context: CompressionContext, model_dir: Path) -> Path:
        if hasattr(context.model, "save_pretrained"):
            context.model.save_pretrained(
                model_dir,
                safe_serialization=True,
                max_shard_size=context.config.checkpoint_shard_size,
            )
            if context.tokenizer is not None and hasattr(context.tokenizer, "save_pretrained"):
                context.tokenizer.save_pretrained(model_dir)
            return model_dir

        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in context.model.state_dict().items()
            if isinstance(tensor, torch.Tensor)
        }
        save_file(state_dict, str(model_dir / "model.safetensors"))
        return model_dir
