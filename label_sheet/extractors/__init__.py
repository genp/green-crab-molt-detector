"""Extractor registry.

Maps the ``config.YearConfig.extractor`` string to a concrete extractor
function. Every extractor reads only from ``data/raw`` (folder names, raw Excel
workbooks, raw ``.docx``); none depend on pre-existing ``data/processed`` labels.
Add a new year by writing a module here and registering it below.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ..config import YearConfig
from ..records import ImageRecord
from . import crate_raw, folder_years, year_2026

_REGISTRY: Dict[str, Callable[[YearConfig], List[ImageRecord]]] = {
    "folder_year": folder_years.extract,
    "crate_docx": crate_raw.extract,
    "year_2026": year_2026.extract,
}


def get_extractor(key: str) -> Callable[[YearConfig], List[ImageRecord]]:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown extractor {key!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
