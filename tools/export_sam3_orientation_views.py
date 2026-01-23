#!/usr/bin/env python3
"""
Export SAM3-oriented training images into three folders for manual review:
    data/sam3_orientation/ventral
    data/sam3_orientation/dorsal
    data/sam3_orientation/uncertain

Uses:
- data/processed/manifest_with_2016_labels.csv
- models/sam3_ventral_metadata.json        (indices of labeled SAM3 crops)
- models/sam3_ventral_filtered_metadata.json (orientation_labels for all 316)

Each copied file is prefixed with its manifest index to avoid name collisions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "data" / "processed" / "manifest_with_2016_labels.csv"
    sam3_meta_path = root / "models" / "sam3_ventral_metadata.json"
    filt_meta_path = root / "models" / "sam3_ventral_filtered_metadata.json"

    df = pd.read_csv(manifest_path)
    df = df[df["days_until_molt"].notna()].copy()

    sam3_meta = json.loads(sam3_meta_path.read_text())
    filt_meta = json.loads(filt_meta_path.read_text())

    indices_all = sam3_meta["indices"]
    orientation_labels = filt_meta["orientation_labels"]

    if len(indices_all) != len(orientation_labels):
        raise RuntimeError(
            f"Length mismatch: {len(indices_all)} indices vs {len(orientation_labels)} orientation labels"
        )

    out_root = root / "data" / "sam3_orientation"
    out_dirs = {
        1: out_root / "ventral",
        0: out_root / "dorsal",
        -1: out_root / "uncertain",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    counts = {1: 0, 0: 0, -1: 0}
    for idx, label in zip(indices_all, orientation_labels):
        label = int(label)
        if label not in out_dirs:
            continue
        row = df.loc[idx]
        src = Path(row["image_path"])
        if not src.exists():
            # Skip missing files rather than crashing.
            continue
        dest_dir = out_dirs[label]
        dest = dest_dir / f"{idx:05d}_{src.name}"
        if not dest.exists():
            shutil.copy2(src, dest)
        counts[label] += 1

    print("Copied images per bucket:")
    print(f"  ventral   (label=1): {counts[1]}")
    print(f"  dorsal    (label=0): {counts[0]}")
    print(f"  uncertain (label=-1): {counts[-1]}")


if __name__ == "__main__":
    main()

