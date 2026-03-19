from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import requests


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / math.pow(1024, 3)


def zip_directory(path: Path) -> Path:
    archive_path = shutil.make_archive(str(path), "zip", root_dir=path)
    return Path(archive_path)


def download_text(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    return destination
