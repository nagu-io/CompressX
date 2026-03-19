from __future__ import annotations

from pathlib import Path

from compressx.utils.imports import optional_import

_FALLBACK_TEXT = [
    "Language models compress knowledge into parameters and activations.",
    "Quantization trades off precision for memory footprint and throughput.",
    "Pruning removes redundant structure while preserving important behavior.",
    "Distillation can recover quality after an aggressive compression step.",
]


def load_text_samples(
    source: Path | None,
    *,
    sample_count: int,
    split: str = "test",
) -> list[str]:
    if source:
        lines = [
            line.strip()
            for line in Path(source).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines[:sample_count] or _FALLBACK_TEXT[:sample_count]

    datasets = optional_import("datasets")
    if datasets is None:
        return _FALLBACK_TEXT[:sample_count]

    try:
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [row["text"].strip() for row in dataset if row.get("text", "").strip()]
        return texts[:sample_count] or _FALLBACK_TEXT[:sample_count]
    except Exception:
        return _FALLBACK_TEXT[:sample_count]
