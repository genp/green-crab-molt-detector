"""
Paths, per-year configuration, and the model registry.

Everything environment- or choice-specific lives here so the rest of the package
stays declarative. Edit ``YEARS`` to add/remove worksheets and ``MODELS`` to swap
which estimators populate the ``m1_``/``m2_`` column groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Repo paths                                                                    #
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

#: Working directory for the pipeline's intermediate artifacts.
WORK_DIR = PROCESSED_DIR / "label_sheet"
#: Records are stored as CSV (no parquet engine is installed in venv/).
RECORDS_PATH = WORK_DIR / "records.csv"
PREDICTIONS_PATH = WORK_DIR / "predictions.csv"
THUMB_CACHE_DIR = WORK_DIR / "thumbnails"

#: Final deliverable.
WORKBOOK_PATH = PROCESSED_DIR / "green_crab_label_sheet.xlsx"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".heic")


@dataclass(frozen=True)
class YearConfig:
    """Configuration for one worksheet (one study year / collection)."""

    sheet_name: str            # worksheet tab name, e.g. "2016"
    extractor: str             # key into extractors.get_extractor
    raw_subdir: Optional[str]  # folder under data/raw, or None if CSV-sourced
    #: Substring(s) that, if present in an image path, mark it as very likely
    #: part of a model's training data (for the *_in_training_set flag).
    training_markers: Sequence[str] = field(default_factory=tuple)


#: The worksheets to build, in tab order. Keep every year even when its labels
#: are sparse — the request requires identical formatting across all years.
YEARS: List[YearConfig] = [
    YearConfig("2016", "folder_year", "NH Green Crab Project 2016",
               training_markers=("NH Green Crab Project 2016",)),
    YearConfig("2017", "folder_year", "NH Green Crab Project -Doyle Fellowship 2017"),
    YearConfig("2018", "folder_year", "2018 NH Green Crab-Doyle Fellowship"),
    # The crate study is the 2019 experiment: "Salinity and Temp Monitoring
    # Crate 1.docx" records 6/21/2019..7/8/2019 and Observations.docx says the
    # experiment began June 19th. Tab kept distinct from the historical years.
    YearConfig("2019_crate", "crate_docx", None,
               training_markers=("crate_images",)),
    YearConfig("2026", "year_2026", "Green Crab AI 2026"),
]


@dataclass(frozen=True)
class ModelConfig:
    """Declares one estimator that fills a model column group.

    Attributes:
        model_id: "m1" or "m2"; must match the schema column prefix.
        display_name: Human name shown in the provenance comment.
        runner: Key into models.registry.get_runner.
        feature_extractor: "vit" or "openclip".
        weights: Path to the joblib regressor (relative to MODELS_DIR).
        bootstrap: If True, produce a bootstrap std via tree resampling.
        trained_on: Sheet names whose data was (likely) in this model's train
            set, used to set *_in_training_set. Best-effort documentation.
        notes: Extra text appended to the provenance comment.
    """

    model_id: str
    display_name: str
    runner: str
    feature_extractor: str
    weights: str
    bootstrap: bool = False
    #: Optional path (relative to MODELS_DIR) to a StandardScaler joblib applied
    #: before the regressor, for artifacts saved as a bare estimator whose scaler
    #: lives in a separate file (e.g. the OpenCLIP regressor).
    scaler: str = ""
    trained_on: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


#: The two models the request asks for. Swap freely; the writer reads these to
#: build the per-column-group provenance comments.
MODELS: List[ModelConfig] = [
    ModelConfig(
        model_id="m1",
        display_name="Primary ViT molt estimator (deployed app model)",
        runner="estimator",
        feature_extractor="vit",
        weights="molt_regressor_vit_random_forest.joblib",
        bootstrap=False,
        trained_on=("2016",),
        notes="torchvision vit_b_16 (768-d) features + RandomForest regressor. "
              "This is the model app_fastapi.py loads by default.",
    ),
    ModelConfig(
        model_id="m2",
        display_name="OpenCLIP ViT-H-14 molt estimator (bootstrap)",
        runner="estimator",
        feature_extractor="openclip",
        weights="openclip_regressor.joblib",
        scaler="openclip_regressor_scaler.joblib",
        bootstrap=True,
        trained_on=("2016",),
        notes="OpenCLIP ViT-H-14/laion2b_s32b_b79k image embedding (L2-normalized, "
              "1024-d) -> StandardScaler (openclip_regressor_scaler.joblib) -> "
              "800-tree RandomForest (openclip_regressor.joblib), matching "
              "tools/run_openclip_regression.py. 'Bootstrap' = per-tree prediction "
              "dispersion reported as m2_days_to_molt_std.",
    ),
]


def get_model(model_id: str) -> ModelConfig:
    for m in MODELS:
        if m.model_id == model_id:
            return m
    raise KeyError(f"No ModelConfig with id {model_id!r}")


def run_timestamp() -> str:
    """UTC timestamp string used in provenance comments."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def ensure_work_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
