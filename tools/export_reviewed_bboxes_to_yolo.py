#!/usr/bin/env python3
"""
Export accepted reviewed bbox proposals to a YOLO detection dataset.

Accepted rows are those with review_status in {accept, accepted, keep}.
Splits are deterministic:
- Rows whose image_path contains any --force-test-substr are assigned to test.
- All others are split by SHA1(image_path) into train/val/test using the provided
  fractions, so another researcher with the same CSV gets the same split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageOps


ACCEPTED = {"accept", "accepted", "keep"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True, help="Reviewed proposals CSV.")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output directory.")
    parser.add_argument("--dataset-yaml", type=Path, default=None, help="Optional YAML output path.")
    parser.add_argument("--force-test-substr", action="append", default=[], help="Substring in image_path forced to test.")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--class-name", default="green_crab")
    return parser.parse_args()


def read_accepted(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("review_status", "").strip().lower() in ACCEPTED]


def split_for_image(path: str, force_test_substr: Iterable[str], val_frac: float, test_frac: float) -> str:
    if any(substr and substr in path for substr in force_test_substr):
        return "test"
    value = int(hashlib.sha1(path.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < test_frac:
        return "test"
    if value < test_frac + val_frac:
        return "val"
    return "train"


def yolo_line(row: Dict[str, str], image_w: int, image_h: int) -> str:
    x1 = float(row["bbox_xmin"])
    y1 = float(row["bbox_ymin"])
    x2 = float(row["bbox_xmax"])
    y2 = float(row["bbox_ymax"])
    cx = ((x1 + x2) / 2) / image_w
    cy = ((y1 + y2) / 2) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def copy_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.copy2(src, dest)


def main() -> None:
    args = parse_args()
    rows = read_accepted(args.reviewed)
    if not rows:
        raise SystemExit(f"No accepted rows found in {args.reviewed}")

    by_image: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)

    counts = defaultdict(int)
    for image_path_str, image_rows in sorted(by_image.items()):
        image_path = Path(image_path_str)
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue
        split = split_for_image(image_path_str, args.force_test_substr, args.val_frac, args.test_frac)
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            image_w, image_h = image.size
        image_dest = args.output / "images" / split / image_path.name
        label_dest = args.output / "labels" / split / f"{image_path.stem}.txt"
        copy_image(image_path, image_dest)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.write_text("\n".join(yolo_line(row, image_w, image_h) for row in image_rows) + "\n")
        counts[split] += 1

    yaml_path = args.dataset_yaml or args.output.with_suffix(".yaml")
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        )
    )

    print(f"Accepted boxes: {len(rows)}")
    print(f"Images by split: {dict(counts)}")
    print(f"Wrote YOLO dataset: {args.output}")
    print(f"Wrote YAML: {yaml_path}")


if __name__ == "__main__":
    main()
