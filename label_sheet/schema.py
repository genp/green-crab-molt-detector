"""
Canonical column schema for the centralized green crab label sheet.

This module is the *single source of truth* for the worksheet layout. Every
year's worksheet uses exactly these columns in exactly this order, so that a
year missing some data still lines up with every other year. The extractors, the
model runners, and the XLSX writer all import from here; nothing else should
hard-code column names.

The controlled vocabularies intentionally mirror ``field_data/SPREADSHEET_README.md``
so that the new workbook is parseable by the same downstream tooling that already
understands the field-data spreadsheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Controlled vocabularies (kept identical to field_data/SPREADSHEET_README.md)  #
# --------------------------------------------------------------------------- #

MOLT_PHASES: Sequence[str] = (
    "intermolt",
    "pre_molt",
    "peeler_imminent",
    "molted",
    "dead",
    "unknown",
)

VIEWS: Sequence[str] = ("ventral", "dorsal", "side", "unknown")

SEXES: Sequence[str] = ("male", "female", "unknown")

CONFIDENCES: Sequence[str] = ("high", "medium", "low", "unknown")

ESTIMATE_INPUTS: Sequence[str] = ("yolo_crop", "whole_image_fallback", "not_run")

EXPERT_STATUSES: Sequence[str] = ("", "confirm", "reject", "update")


@dataclass(frozen=True)
class Column:
    """One worksheet column.

    Attributes:
        key: Stable identifier used as the ``ImageRecord`` field name and the
            key in the flat record dictionaries. Never shown to the user.
        header: Human-facing header text written into row 1.
        group: Logical group ("identity", "ground_truth", "aux", "model_m1",
            "model_m2", "expert"). Used for coloring and for attaching the
            per-group model-provenance comment.
        width: Column width in Excel units.
        is_image: True only for the single thumbnail column (rendered as an
            embedded picture rather than text).
        dropdown: Optional controlled vocabulary; when set the writer adds an
            Excel data-validation dropdown to the column body.
        wrap: Whether to wrap long text in the cell.
    """

    key: str
    header: str
    group: str
    width: float = 16.0
    is_image: bool = False
    dropdown: Optional[Sequence[str]] = None
    wrap: bool = False


# --------------------------------------------------------------------------- #
# The canonical, ordered column list.                                          #
# --------------------------------------------------------------------------- #
# Order matters: column A must be the thumbnail so the row height that fits the
# image also frames the rest of the row.

COLUMNS: List[Column] = [
    # --- Identity / paths -------------------------------------------------- #
    Column("thumbnail", "thumbnail", "identity", width=14.0, is_image=True),
    Column("year", "year", "identity", width=8.0),
    Column("dataset", "dataset", "identity", width=26.0, wrap=True),
    Column("crab_id", "crab_id", "identity", width=12.0),
    Column("observation_id", "observation_id", "identity", width=20.0, wrap=True),
    Column("session_id", "session_id", "identity", width=20.0, wrap=True),
    Column("original_filename", "original_filename", "identity", width=20.0),
    Column("image_relpath", "image_relpath", "identity", width=52.0, wrap=True),
    # --- Ground-truth labels ---------------------------------------------- #
    Column("capture_date", "capture_date", "ground_truth", width=12.0),
    Column("molt_date", "molt_date", "ground_truth", width=12.0),
    Column("days_until_molt", "days_until_molt", "ground_truth", width=12.0),
    Column("known_molt_phase", "known_molt_phase", "ground_truth", width=14.0, dropdown=MOLT_PHASES),
    Column("sex", "sex", "ground_truth", width=8.0, dropdown=SEXES),
    Column("view", "view", "ground_truth", width=9.0, dropdown=VIEWS),
    Column("color_state", "color_state", "ground_truth", width=14.0),
    Column("carapace_width_mm", "carapace_width_mm", "ground_truth", width=12.0),
    Column("outcome", "outcome", "ground_truth", width=12.0),
    Column("label_source", "label_source", "ground_truth", width=22.0, wrap=True),
    Column("label_confidence", "label_confidence", "ground_truth", width=12.0, dropdown=CONFIDENCES),
    Column("source_notes", "source_notes", "ground_truth", width=34.0, wrap=True),
    # --- Aux metadata ------------------------------------------------------ #
    Column("condo_id", "condo_id", "aux", width=10.0),
    Column("cell_id", "cell_id", "aux", width=8.0),
    Column("degree_of_molt", "degree_of_molt", "aux", width=12.0),
    Column("is_in_situ", "is_in_situ", "aux", width=9.0),
    # --- Model A: primary estimator (key prefix m1_; shown as ViT_) -------- #
    # NOTE: the ``key`` (m1_*) is the stable pipeline identifier; ``header``
    # (ViT_*) is only the display text reviewers see.
    Column("m1_days_to_molt", "ViT_days_to_molt", "model_m1", width=13.0),
    Column("m1_phase", "ViT_phase", "model_m1", width=13.0, dropdown=MOLT_PHASES),
    Column("m1_confidence", "ViT_confidence", "model_m1", width=12.0, dropdown=CONFIDENCES),
    Column("m1_sex", "ViT_sex", "model_m1", width=9.0, dropdown=SEXES),
    Column("m1_view", "ViT_view", "model_m1", width=9.0, dropdown=VIEWS),
    Column("m1_molt_indicators", "ViT_molt_indicators", "model_m1", width=30.0, wrap=True),
    Column("m1_estimate_input", "ViT_estimate_input", "model_m1", width=16.0, dropdown=ESTIMATE_INPUTS),
    Column("m1_in_training_set", "ViT_in_training_set", "model_m1", width=13.0),
    # --- Model B: bootstrap estimator (key prefix m2_; shown as OpenCLIP_) -- #
    Column("m2_days_to_molt", "OpenCLIP_days_to_molt", "model_m2", width=13.0),
    Column("m2_days_to_molt_std", "OpenCLIP_days_to_molt_std", "model_m2", width=16.0),
    Column("m2_phase", "OpenCLIP_phase", "model_m2", width=13.0, dropdown=MOLT_PHASES),
    Column("m2_confidence", "OpenCLIP_confidence", "model_m2", width=12.0, dropdown=CONFIDENCES),
    Column("m2_sex", "OpenCLIP_sex", "model_m2", width=9.0, dropdown=SEXES),
    Column("m2_view", "OpenCLIP_view", "model_m2", width=9.0, dropdown=VIEWS),
    Column("m2_molt_indicators", "OpenCLIP_molt_indicators", "model_m2", width=30.0, wrap=True),
    Column("m2_estimate_input", "OpenCLIP_estimate_input", "model_m2", width=16.0, dropdown=ESTIMATE_INPUTS),
    Column("m2_in_training_set", "OpenCLIP_in_training_set", "model_m2", width=13.0),
    # --- Expert review (prefix expert_) ------------------------------------ #
    Column("expert_status", "expert_status", "expert", width=12.0, dropdown=EXPERT_STATUSES),
    Column("expert_days_to_molt", "expert_days_to_molt", "expert", width=15.0),
    Column("expert_molt_phase", "expert_molt_phase", "expert", width=15.0, dropdown=MOLT_PHASES),
    Column("expert_sex", "expert_sex", "expert", width=10.0, dropdown=SEXES),
    Column("expert_view", "expert_view", "expert", width=10.0, dropdown=VIEWS),
    Column("expert_molt_indicators", "expert_molt_indicators", "expert", width=28.0, wrap=True),
    Column("expert_notes", "expert_notes", "expert", width=34.0, wrap=True),
]

# Convenience lookups -------------------------------------------------------- #

COLUMN_KEYS: List[str] = [c.key for c in COLUMNS]

#: Keys that hold real data (everything except the rendered thumbnail image).
DATA_KEYS: List[str] = [c.key for c in COLUMNS if not c.is_image]


def columns_in_group(group: str) -> List[Column]:
    """Return the columns belonging to ``group`` in worksheet order."""
    return [c for c in COLUMNS if c.group == group]


def column_index(key: str) -> int:
    """Return the 1-based worksheet column index for ``key`` (A == 1)."""
    return COLUMN_KEYS.index(key) + 1


#: Fill colors (ARGB) per group, used to visually separate the column bands.
GROUP_FILLS = {
    "identity": "FFEFEFEF",     # light gray
    "ground_truth": "FFDDEBF7",  # light blue
    "aux": "FFE2EFDA",           # light green
    "model_m1": "FFFCE4D6",      # light orange
    "model_m2": "FFFFF2CC",      # light yellow
    "expert": "FFEAD1DC",        # light plum
}
