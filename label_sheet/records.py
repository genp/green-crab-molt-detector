"""
The normalized per-image record that flows through the whole pipeline.

Extractors produce ``ImageRecord`` objects (ground-truth + metadata only). The
predict stage fills the ``m1_*`` / ``m2_*`` fields. The assemble stage reads the
records back and writes one worksheet row per record.

Records are serialized to a flat parquet/CSV of ``DATA_KEYS`` columns so the
three stages stay decoupled and independently runnable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .schema import DATA_KEYS


@dataclass
class ImageRecord:
    """One image and everything known about it.

    Only ground-truth / metadata fields are populated by extractors. Model
    fields (``m1_*``, ``m2_*``) and expert fields are left blank here and filled
    downstream. Every attribute name matches a key in ``schema.DATA_KEYS`` so
    ``to_row`` can round-trip losslessly.
    """

    # Identity / paths
    year: str
    dataset: str
    image_relpath: str
    crab_id: str = ""
    observation_id: str = ""
    session_id: str = ""
    original_filename: str = ""

    # Ground-truth labels
    capture_date: str = ""
    molt_date: str = ""
    days_until_molt: str = ""
    known_molt_phase: str = ""
    sex: str = ""
    view: str = ""
    color_state: str = ""
    carapace_width_mm: str = ""
    outcome: str = ""
    label_source: str = ""
    label_confidence: str = ""
    source_notes: str = ""

    # Aux metadata
    condo_id: str = ""
    cell_id: str = ""
    degree_of_molt: str = ""
    is_in_situ: str = ""

    # Absolute path is carried for the predict/thumbnail stages but is NOT a
    # worksheet column (the sheet shows image_relpath). Excluded from to_row.
    abs_path: str = ""

    def to_row(self) -> Dict[str, Any]:
        """Return a dict with exactly the writable schema keys.

        Model and expert columns default to empty string so the flat table has a
        stable width even before the predict stage runs.
        """
        row: Dict[str, Any] = {key: "" for key in DATA_KEYS}
        for key in DATA_KEYS:
            if hasattr(self, key):
                row[key] = getattr(self, key)
        return row


def records_to_frame(records: List[ImageRecord]) -> pd.DataFrame:
    """Build a DataFrame of records, preserving ``abs_path`` as an extra column.

    ``abs_path`` is needed by the predict + thumbnail stages but is dropped
    before writing the workbook.
    """
    rows = []
    for rec in records:
        row = rec.to_row()
        row["abs_path"] = rec.abs_path
        rows.append(row)
    frame = pd.DataFrame(rows, columns=[*DATA_KEYS, "abs_path"])
    return frame


def normalize_str(value: Any) -> str:
    """Coerce any cell value to a clean string ('' for missing/NaN)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none"}:
        return ""
    return text


def relpath(abs_path: Path, root: Path) -> str:
    """Return ``abs_path`` relative to ``root`` using POSIX separators.

    Falls back to the absolute string if the path is outside the repo (which can
    happen for symlinked datasets), so the sheet always has a usable reference.
    """
    try:
        return abs_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return abs_path.as_posix()
