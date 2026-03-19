from __future__ import annotations

from abc import ABC, abstractmethod

from compressx.context import CompressionContext


class PipelineStage(ABC):
    name = "stage"

    @abstractmethod
    def run(self, context: CompressionContext) -> None:
        raise NotImplementedError
