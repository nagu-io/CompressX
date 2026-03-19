"""Pruning entry point for CompressX."""

from compressx.stages.pruning import StructuralPrunerStage

StructuralPruner = StructuralPrunerStage

__all__ = ["StructuralPruner"]
