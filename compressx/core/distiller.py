"""Distillation entry point for CompressX."""

from compressx.stages.distillation import DistillationFineTunerStage

DistillationFineTuner = DistillationFineTunerStage

__all__ = ["DistillationFineTuner"]
