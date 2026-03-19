from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
