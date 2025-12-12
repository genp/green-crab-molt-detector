"""
Merge crate .docx-derived data into the main crab dataset to expand labels.

Inputs:
- data/processed/crab_dataset.csv (2016 loader output)
- data/processed/crate_docs.csv (from extract_crate_docx.py)

Outputs:
- data/processed/crab_dataset_merged.csv
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import re
from datetime import datetime
from typing import List, Tuple


def parse_dates(paras: str, year: int = 2016) -> List[datetime]:
    dates = []
    for part in re.split(r"[;\s]+", paras):
        if not part.strip():
            continue
        m = re.match(r"(\\d{1,2})/(\\d{1,2})", part)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            dates.append(datetime(year, month, day))
    return dates


def expand_crate_rows(crate_df: pd.DataFrame) -> pd.DataFrame:
    # Map degree of molt to approximate days until molt (heuristic)
    degree_map = {
        "imminent": 0.5,  # tighter window for imminent
        "late": 4.0,
        "mid": 9.0,
        "new": 16.0,
    }

    rows = []
    for _, row in crate_df.iterrows():
        dates = parse_dates(row["paragraphs"], year=2016)
        images = str(row["image_paths"]).split("|") if pd.notna(row["image_paths"]) else []
        degree = str(row.get("degree_of_molt", "")).strip().lower()
        approx_days = None
        for key, val in degree_map.items():
            if key in degree:
                approx_days = val
                break
        for i, img in enumerate(images):
            capture_date = dates[min(i, len(dates) - 1)] if dates else None
            rows.append(
                {
                    "crab_id": f"Crate{row['crate']}_{row['crab_number']}",
                    "sex": None,
                    "capture_date": capture_date,
                    "molt_date": None,
                    "days_until_molt": approx_days,
                    "is_molted": False,
                    "image_path": img,
                    "source_folder": row["crate"],
                }
            )
    return pd.DataFrame(rows)


def main():
    base = Path("data/processed")
    main_df = pd.read_csv(base / "crab_dataset.csv")
    crate_df = pd.read_csv(base / "crate_docs.csv")

    expanded_crate = expand_crate_rows(crate_df)

    merged = pd.concat([main_df, expanded_crate], ignore_index=True)
    merged.to_csv(base / "crab_dataset_merged.csv", index=False)
    print(
        f"Merged dataset saved to {base / 'crab_dataset_merged.csv'} "
        f"rows={len(merged)} labeled={merged['days_until_molt'].notna().sum()}"
    )


if __name__ == "__main__":
    main()
