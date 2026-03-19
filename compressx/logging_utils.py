from __future__ import annotations

import logging
from pathlib import Path


class _StageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = "GENERAL"
        return True


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compressx")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.filters.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(stage)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    stage_filter = _StageFilter()
    logger.addFilter(stage_filter)
    file_handler.addFilter(stage_filter)
    stream_handler.addFilter(stage_filter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_stage_logger(logger: logging.Logger, stage: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"stage": stage.upper()})
