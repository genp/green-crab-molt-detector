"""
Build a manifest for the green crab dataset.

The script walks data/raw, extracts metadata (crab_id, sex, capture_date,
molt_date, days_until_molt, outcome, session_id, image_path), and writes:
 - data/processed/manifest.csv
 - data/processed/manifest_summary.json (basic counts and label coverage)

It is designed to accommodate the mixed folder conventions across 2016/2017/2018
datasets (crate names with sex hints, molt dates in folder names, death/throwback
tags). This keeps the regression target (days_until_molt) intact for retraining
ViT and temporal models.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

RE_MOLT_TAG = re.compile(r"molted[ _-]*([0-9]{1,2})[.:/-]?([0-9]{1,2})", re.IGNORECASE)
RE_CRAB_ID = re.compile(r"([MF])([0-9]+)", re.IGNORECASE)
RE_DATE_INLINE = re.compile(r"([0-9]{1,2})[./-]([0-9]{1,2})[./-]?([0-9]{2,4})?")
RE_GENERIC_ID = re.compile(r"([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{1,2,4}[- _]?[0-9]*)")

VALID_YEARS = {2016, 2017, 2018}


@dataclass
class Record:
    crab_id: Optional[str]
    sex: Optional[str]
    capture_date: Optional[datetime]
    molt_date: Optional[datetime]
    days_until_molt: Optional[float]
    outcome: Optional[str]
    session_id: str
    image_path: str
    source_folder: str


def parse_year_from_root(root: Path) -> int:
    """Infer the study year from the top-level directory name."""
    name = root.name
    if name.startswith("NH Green Crab Project 2016"):
        return 2016
    if "2017" in name:
        return 2017
    if "2018" in name:
        return 2018
    # Default to 2016 if unknown
    return 2016


def is_year_allowed(year: int, allowed_years: set[int]) -> bool:
    return year in allowed_years


def normalize_sex_from_crate(path: Path) -> Optional[str]:
    """Infer sex from crate/parent folder names."""
    name = path.name.lower()
    if "male" in name or "(m)" in name or "males" in name:
        return "M"
    if "female" in name or "females" in name or "(f)" in name or "egg" in name:
        return "F"
    return None


def parse_crab_id(folder: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Parse crab_id and molt_date from folder names like:
    - F2 (molted 9:20)
    - M3 (molted 9:19)
    - 7-11-18-03 (no explicit ID, but has an inline date)
    """
    # Direct crab id with molt tag
    molt_match = RE_MOLT_TAG.search(folder)
    id_match = RE_CRAB_ID.search(folder)
    crab_id = id_match.group(0) if id_match else None
    molt_date = None
    if molt_match:
        # Year is injected later based on root
        month = int(molt_match.group(1))
        day = int(molt_match.group(2))
        molt_date = (month, day)

    sex = None
    if crab_id:
        sex = crab_id[0].upper()

    return crab_id, sex, molt_date


def parse_date_from_str(name: str, default_year: int) -> Optional[datetime]:
    """
    Parse a date from strings like "8:26", "7-11-18-03", "6.28 1 molted 6.29".
    """
    match = RE_DATE_INLINE.search(name)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    year = int(match.group(3)) if match.group(3) else default_year
    # Fix short years (e.g., 18 -> 2018)
    if year < 100:
        year += 2000
    return datetime(year, month, day)


def compute_days_until_molt(capture: Optional[datetime], molt: Optional[datetime]) -> Optional[float]:
    if capture is None or molt is None:
        return None
    return float((molt - capture).days)


def iter_images(root: Path) -> Iterable[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        yield from root.rglob(f"*{ext}")


def build_manifest(base_path: Path, allowed_years: set[int]) -> Tuple[pd.DataFrame, Dict]:
    records: List[Record] = []
    summary: Dict[str, Dict] = defaultdict(dict)

    # Iterate immediate subdirs of data/raw
    for dataset_root in sorted([p for p in base_path.iterdir() if p.is_dir()]):
        year = parse_year_from_root(dataset_root)
        if not is_year_allowed(year, allowed_years):
            continue
        crate_sex_hint = normalize_sex_from_crate(dataset_root)
        files = list(iter_images(dataset_root))
        for img_path in tqdm(files, desc=f"Scanning {dataset_root.name}"):
            parent = img_path.parent
            # Attempt to derive crab_id and molt_date from ancestor names
            ancestors = list(img_path.relative_to(dataset_root).parents)
            ancestors_names = [p.name for p in ancestors]

            crab_id = None
            sex = crate_sex_hint
            molt_date_partial = None
            capture_date = None
            outcome = None

            # Walk up the path parts from leaf upward to find IDs/dates
            for part in ancestors_names:
                cid, csex, mdate_partial = parse_crab_id(part)
                crab_id = crab_id or cid
                sex = sex or csex or normalize_sex_from_crate(Path(part))
                molt_date_partial = molt_date_partial or mdate_partial
                # Outcome tags
                low = part.lower()
                if "died" in low:
                    outcome = outcome or "died"
                elif "thrown back" in low or "thrownback" in low:
                    outcome = outcome or "thrown_back"
                elif "molted" in low:
                    outcome = outcome or "molted"
                # Capture date fallback
                if capture_date is None:
                    parsed = parse_date_from_str(part, year)
                    if parsed:
                        capture_date = parsed

                # Generic ID fallback: if no crab_id, use date-coded folder name
                if crab_id is None:
                    gid = RE_GENERIC_ID.search(part)
                    if gid:
                        crab_id = gid.group(1).replace("/", "-")

            # If no capture_date yet, try the stem of the image (often contains date)
            if capture_date is None:
                capture_date = parse_date_from_str(img_path.stem, year)

            # If still no crab_id, fallback to parent folder name to avoid NaNs
            if crab_id is None:
                crab_id = parent.name

            molt_date = None
            if molt_date_partial:
                month, day = molt_date_partial
                molt_date = datetime(year, month, day)

            days_until_molt = compute_days_until_molt(capture_date, molt_date)

            session_id = f"{dataset_root.name}::{capture_date.date() if capture_date else 'unknown'}"

            rec = Record(
                crab_id=crab_id,
                sex=sex,
                capture_date=capture_date,
                molt_date=molt_date,
                days_until_molt=days_until_molt,
                outcome=outcome,
                session_id=session_id,
                image_path=str(img_path),
                source_folder=str(dataset_root),
            )
            records.append(rec)

        summary[dataset_root.name]["num_images"] = len(files)

    df = pd.DataFrame([asdict(r) for r in records])

    # Basic cleanup: drop obvious non-images and NaT placeholders
    if not df.empty:
        # Coerce dates
        df["capture_date"] = pd.to_datetime(df["capture_date"])
        df["molt_date"] = pd.to_datetime(df["molt_date"])

    # Summaries
    sex_counts = df["sex"].value_counts(dropna=False).to_dict()
    outcome_counts = df["outcome"].value_counts(dropna=False).to_dict()
    known_labels = df["days_until_molt"].notna().sum()
    summary["global"] = {
        "num_images": int(len(df)),
        "num_crabs": int(df["crab_id"].nunique(dropna=True)),
        "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
        "outcome_counts": {str(k): int(v) for k, v in outcome_counts.items()},
        "num_with_labels": int(known_labels),
    }
    return df, summary


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Full manifest (all years)
    df, summary = build_manifest(raw_path, allowed_years=VALID_YEARS)

    manifest_path = processed_path / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    summary_path = processed_path / "manifest_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote manifest: {manifest_path} ({len(df)} rows)")
    print(f"Wrote summary: {summary_path}")
    print("Label coverage: "
          f"{summary['global']['num_with_labels']} with days_until_molt out of {summary['global']['num_images']} images")

    # 2016-only baseline manifest
    df_2016, summary_2016 = build_manifest(raw_path, allowed_years={2016})
    manifest_2016_path = processed_path / "manifest_2016.csv"
    summary_2016_path = processed_path / "manifest_2016_summary.json"
    df_2016.to_csv(manifest_2016_path, index=False)
    with summary_2016_path.open("w") as f:
        json.dump(summary_2016, f, indent=2, default=str)
    print(f"Wrote 2016-only manifest: {manifest_2016_path} ({len(df_2016)} rows)")
    print("2016 label coverage: "
          f"{summary_2016['global']['num_with_labels']} with days_until_molt out of {summary_2016['global']['num_images']} images")


if __name__ == "__main__":
    main()
