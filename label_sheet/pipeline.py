"""
Pipeline orchestration for the three stages: extract, predict, assemble.

Each stage reads/writes flat files under ``config.WORK_DIR`` and can run
independently. ``run_all`` chains them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from . import config
from .config import MODELS, YEARS, ModelConfig, YearConfig, ensure_work_dirs
from .extractors import get_extractor
from .records import records_to_frame
from .schema import DATA_KEYS


# --------------------------------------------------------------------------- #
# Stage 1: extract                                                             #
# --------------------------------------------------------------------------- #

def stage_extract(years: Optional[List[YearConfig]] = None) -> pd.DataFrame:
    """Run every year's extractor and persist the combined records table."""
    ensure_work_dirs()
    years = years or YEARS
    frames: List[pd.DataFrame] = []
    for cfg in years:
        extractor = get_extractor(cfg.extractor)
        records = extractor(cfg)
        frame = records_to_frame(records)
        frame.insert(0, "_sheet", cfg.sheet_name)
        frames.append(frame)
        print(f"[extract] {cfg.sheet_name}: {len(records)} images")
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(config.RECORDS_PATH, index=False)
    print(f"[extract] wrote {len(combined)} records -> {config.RECORDS_PATH}")
    return combined


def load_records() -> pd.DataFrame:
    if config.RECORDS_PATH.exists():
        # Keep every column as string; blanks must stay blank, not NaN/float.
        return pd.read_csv(config.RECORDS_PATH, dtype=str, keep_default_na=False)
    raise FileNotFoundError("No records found; run the extract stage first.")


# --------------------------------------------------------------------------- #
# Stage 2: predict                                                            #
# --------------------------------------------------------------------------- #

def _is_training_image(abs_path: str, cfg: ModelConfig, sheet: str) -> bool:
    """Best-effort: is this image likely in the model's training set?"""
    if sheet in cfg.trained_on:
        return True
    # Also honor per-year training markers in the path.
    for yc in YEARS:
        if yc.sheet_name == sheet:
            return any(m in abs_path for m in yc.training_markers) and sheet in cfg.trained_on
    return False


def stage_predict(models: Optional[List[ModelConfig]] = None,
                  limit: Optional[int] = None,
                  flush_every: int = 50) -> None:
    """Run each model over every image, caching results incrementally."""
    from .models.cache import PredictionCache
    from .models.registry import get_runner

    ensure_work_dirs()
    models = models or MODELS
    records = load_records()
    cache = PredictionCache(config.PREDICTIONS_PATH)

    for cfg in models:
        # Which images still need this model?
        todo = [
            (row["_sheet"], row["image_relpath"], row["abs_path"])
            for _, row in records.iterrows()
            if row.get("abs_path") and not cache.has(row["image_relpath"], cfg.model_id)
        ]
        if limit is not None:
            todo = todo[:limit]
        if not todo:
            print(f"[predict] {cfg.model_id}: nothing to do (all cached)")
            continue
        print(f"[predict] {cfg.model_id} ({cfg.display_name}): {len(todo)} images")

        try:
            runner = get_runner(cfg)
        except Exception as exc:
            print(f"[predict] {cfg.model_id}: runner unavailable ({exc}); skipping")
            continue

        done = 0
        for sheet, rel, abs_path in todo:
            in_train = _is_training_image(abs_path, cfg, sheet)
            try:
                pred = runner.predict_path(abs_path, in_train)
            except Exception as exc:
                from .models.registry import Prediction
                pred = Prediction(confidence="unknown",
                                  in_training_set="true" if in_train else "false")
                print(f"[predict] error on {rel}: {exc}")
            cache.add(rel, cfg.model_id, pred)
            done += 1
            if done % flush_every == 0:
                cache.flush()
                print(f"[predict]   {done}/{len(todo)}")
        cache.flush()
        print(f"[predict] {cfg.model_id}: done ({done})")


# --------------------------------------------------------------------------- #
# Stage 3: assemble                                                           #
# --------------------------------------------------------------------------- #

def stage_assemble(use_models: bool = True) -> None:
    """Join records + prediction cache and write the workbook."""
    from .models.cache import PredictionCache
    from .xlsx_writer import write_workbook

    records = load_records()

    if use_models and config.PREDICTIONS_PATH.exists():
        lookup = PredictionCache(config.PREDICTIONS_PATH).wide_lookup()
    else:
        lookup = {}

    # Fill model columns from the cache.
    for key in DATA_KEYS:
        if key.startswith(("m1_", "m2_")) and key not in records.columns:
            records[key] = ""
    if lookup:
        def fill(row):
            preds = lookup.get(row["image_relpath"], {})
            for k, v in preds.items():
                if k in row.index:
                    row[k] = v
            return row
        records = records.apply(fill, axis=1)

    # Build one frame per sheet in the configured order.
    sheet_frames: Dict[str, pd.DataFrame] = {}
    for cfg in YEARS:
        sub = records[records["_sheet"] == cfg.sheet_name].copy()
        keep = [c for c in [*DATA_KEYS, "abs_path"] if c in sub.columns]
        sheet_frames[cfg.sheet_name] = sub[keep].reset_index(drop=True)

    write_workbook(sheet_frames, MODELS, config.WORKBOOK_PATH)
    total = sum(len(f) for f in sheet_frames.values())
    print(f"[assemble] wrote {len(sheet_frames)} sheets, {total} rows -> "
          f"{config.WORKBOOK_PATH}")


def run_all(use_models: bool = True, limit: Optional[int] = None) -> None:
    stage_extract()
    if use_models:
        stage_predict(limit=limit)
    stage_assemble(use_models=use_models)
