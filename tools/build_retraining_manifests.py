#!/usr/bin/env python3
"""
Build global split, detector, and estimator manifests for retraining.

This is a conservative manifest builder. It does not invent molt labels or bbox
labels; it normalizes existing reviewed detector proposals and existing
timing-labeled estimator rows into split-aware manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ACCEPTED_BBOX_STATUS = {"accept", "accepted", "keep"}
ALLOWED_SPLITS = {"train", "val", "test", "field_qa_holdout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-reviewed", action="append", type=Path, default=[])
    parser.add_argument("--estimator-source", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--field-qa-substr", action="append", default=[])
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_path(path: str) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser())


def source_dataset_for_path(image_path: str) -> str:
    parts = Path(image_path).parts
    if "data" in parts and "raw" in parts:
        raw_idx = parts.index("raw")
        if raw_idx + 1 < len(parts):
            return parts[raw_idx + 1]
    if "data" in parts and "processed" in parts:
        processed_idx = parts.index("processed")
        if processed_idx + 1 < len(parts):
            return f"processed/{parts[processed_idx + 1]}"
    parent = Path(image_path).parent
    return parent.name or "unknown"


def infer_view(*values: str) -> str:
    text = " ".join(value or "" for value in values).lower()
    if "ventral" in text or "underside" in text:
        return "ventral"
    if "dorsal" in text or "shell" in text:
        return "dorsal"
    if "side" in text:
        return "side"
    return "unknown"


def infer_color_state(*values: str) -> str:
    text = " ".join(value or "" for value in values).lower()
    if "red" in text:
        return "red_green_crab"
    if "orange" in text or "yellow" in text:
        return "red_green_crab"
    if "green" in text:
        return "normal_green"
    return "unknown"


def infer_in_situ(*values: str) -> str:
    text = " ".join(value or "" for value in values).lower()
    terms = ("in situ", "insitu", "foraging", "mud", "rock", "underwater", "field")
    return "true" if any(term in text for term in terms) else "false"


def infer_negative_type(*values: str) -> str:
    text = " ".join(value or "" for value in values).lower()
    for term in ("human", "glove", "hand", "equipment", "cooler", "crate", "rock"):
        if term in text:
            return "human" if term in {"human", "hand"} else term
    return ""


def stable_hash(value: str) -> float:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_for_group(group_id: str, rows: Iterable[Dict[str, object]], val_frac: float, test_frac: float) -> str:
    row_list = list(rows)
    joined_paths = " ".join(str(row.get("image_path", "")) for row in row_list)
    if any(str(row.get("split", "")) == "field_qa_holdout" for row in row_list):
        return "field_qa_holdout"

    # Preserve hard-case coverage with deterministic buckets when possible.
    row_tags = []
    for row in row_list:
        row_tags.extend(
            [
                str(row.get("color_state", "")),
                str(row.get("is_in_situ", "")),
                str(row.get("view", "")),
                str(row.get("negative_type", "")),
            ]
        )
    tags = " ".join(row_tags + [joined_paths]).lower()
    value = stable_hash(group_id)
    if any(tag in tags for tag in ("red_green_crab", "true", "side", "human", "glove", "equipment")):
        return "test" if value < max(test_frac, 0.20) else "train"
    if value < test_frac:
        return "test"
    if value < test_frac + val_frac:
        return "val"
    return "train"


def detector_group_id(row: Dict[str, str]) -> str:
    image_path = normalize_path(row.get("image_path", ""))
    source_dataset = source_dataset_for_path(image_path)
    return f"{source_dataset}::image::{Path(image_path).stem}"


def estimator_group_id(row: Dict[str, str]) -> str:
    image_path = normalize_path(row.get("image_path", ""))
    source_dataset = source_dataset_for_path(image_path)
    crab_id = (row.get("crab_id") or "").strip()
    session_id = (row.get("session_id") or "").strip()
    if crab_id:
        return f"{source_dataset}::crab::{crab_id}"
    if session_id:
        return f"{source_dataset}::session::{session_id}"
    return f"{source_dataset}::image::{Path(image_path).stem}"


def build_detector_rows(paths: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()
    for path in paths:
        for row in read_csv(path):
            status = (row.get("review_status") or "").strip().lower()
            image_path = normalize_path(row.get("image_path", ""))
            key = (image_path, row.get("bbox_xmin"), row.get("bbox_ymin"), row.get("bbox_xmax"), row.get("bbox_ymax"))
            if key in seen:
                continue
            seen.add(key)
            notes = row.get("review_notes", "")
            is_negative = status not in ACCEPTED_BBOX_STATUS
            rows.append(
                {
                    "image_path": image_path,
                    "source_group_id": detector_group_id(row),
                    "split": "",
                    "bbox_xmin": row.get("bbox_xmin", ""),
                    "bbox_ymin": row.get("bbox_ymin", ""),
                    "bbox_xmax": row.get("bbox_xmax", ""),
                    "bbox_ymax": row.get("bbox_ymax", ""),
                    "species": "green_crab" if not is_negative else "",
                    "view": infer_view(image_path, row.get("prompt", ""), notes),
                    "color_state": infer_color_state(image_path, notes),
                    "is_in_situ": infer_in_situ(image_path, notes),
                    "is_negative": str(is_negative).lower(),
                    "negative_type": infer_negative_type(image_path, notes) if is_negative else "",
                    "label_source": f"reviewed_sam3:{path}",
                    "label_confidence": "high" if status in ACCEPTED_BBOX_STATUS else "reviewed_negative",
                    "review_status": status,
                    "notes": notes,
                }
            )
    return rows


def build_estimator_rows(paths: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()
    for path in paths:
        for row in read_csv(path):
            days = (row.get("days_to_molt") or row.get("days_until_molt") or "").strip()
            image_path = normalize_path(row.get("image_path", ""))
            if not days or not image_path:
                continue
            key = (image_path, days)
            if key in seen:
                continue
            seen.add(key)
            color_state = infer_color_state(image_path, row.get("color", ""))
            rows.append(
                {
                    "image_path": image_path,
                    "source_group_id": estimator_group_id(row),
                    "split": "",
                    "crab_id": row.get("crab_id", ""),
                    "capture_date": row.get("capture_date", ""),
                    "molt_date": row.get("molt_date", ""),
                    "days_to_molt": days,
                    "species": "green_crab",
                    "view": infer_view(image_path),
                    "sex": row.get("sex", ""),
                    "color_state": color_state,
                    "is_in_situ": infer_in_situ(image_path),
                    "crop_source": "whole_image",
                    "bbox_xmin": "",
                    "bbox_ymin": "",
                    "bbox_xmax": "",
                    "bbox_ymax": "",
                    "label_confidence": "medium",
                    "notes": f"from {path}",
                }
            )
    return rows


def apply_field_qa(rows: List[Dict[str, object]], substrings: List[str]) -> None:
    for row in rows:
        image_path = str(row.get("image_path", ""))
        if any(substr and substr in image_path for substr in substrings):
            row["split"] = "field_qa_holdout"


def assign_global_splits(
    detector_rows: List[Dict[str, object]],
    estimator_rows: List[Dict[str, object]],
    val_frac: float,
    test_frac: float,
) -> Dict[str, str]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in detector_rows + estimator_rows:
        grouped[str(row["source_group_id"])].append(row)
    splits = {
        group_id: split_for_group(group_id, rows, val_frac=val_frac, test_frac=test_frac)
        for group_id, rows in sorted(grouped.items())
    }
    enforce_train_test_coverage(splits, grouped)
    return splits


def group_has_tag(rows: List[Dict[str, object]], column: str, value: str) -> bool:
    return any(str(row.get(column, "")).strip().lower() == value for row in rows)


def enforce_train_test_coverage(splits: Dict[str, str], grouped: Dict[str, List[Dict[str, object]]]) -> None:
    """Guarantee train/test coverage for important hard-case groups when possible."""
    checks = [
        ("color_state", "red_green_crab"),
        ("is_in_situ", "true"),
        ("view", "side"),
        ("negative_type", "human"),
        ("negative_type", "glove"),
        ("negative_type", "equipment"),
    ]
    for column, value in checks:
        group_ids = [gid for gid, rows in grouped.items() if group_has_tag(rows, column, value)]
        non_holdout = [gid for gid in group_ids if splits.get(gid) != "field_qa_holdout"]
        if len(non_holdout) < 2:
            continue
        split_values = {splits[gid] for gid in non_holdout}
        ordered = sorted(non_holdout, key=stable_hash)
        if "train" not in split_values:
            splits[ordered[-1]] = "train"
        if "test" not in split_values:
            splits[ordered[0]] = "test"


def build_registry_rows(detector_rows: List[Dict[str, object]], estimator_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in detector_rows + estimator_rows:
        by_group[str(row["source_group_id"])].append(row)

    registry_rows: List[Dict[str, object]] = []
    for group_id, rows in sorted(by_group.items()):
        first = rows[0]
        registry_rows.append(
            {
                "source_group_id": group_id,
                "split": first.get("split", ""),
                "split_reason": "deterministic_stratified_global",
                "source_dataset": source_dataset_for_path(str(first.get("image_path", ""))),
                "crab_id": first.get("crab_id", ""),
                "session_id": "",
                "collection_date": "",
                "is_in_situ": "true" if any(str(row.get("is_in_situ", "")).lower() == "true" for row in rows) else "false",
                "color_state": next((str(row.get("color_state", "")) for row in rows if row.get("color_state")), "unknown"),
                "view": next((str(row.get("view", "")) for row in rows if row.get("view")), "unknown"),
                "species": next((str(row.get("species", "")) for row in rows if row.get("species")), "unknown"),
                "has_detector_label": str(any("bbox_xmin" in row and row.get("label_source") for row in rows)).lower(),
                "has_estimator_label": str(any(row.get("days_to_molt") for row in rows)).lower(),
                "has_negative_label": str(any(str(row.get("is_negative", "")).lower() == "true" for row in rows)).lower(),
                "negative_type": next((str(row.get("negative_type", "")) for row in rows if row.get("negative_type")), ""),
                "notes": "",
            }
        )
    return registry_rows


def main() -> int:
    args = parse_args()
    detector_rows = build_detector_rows(args.detector_reviewed)
    estimator_rows = build_estimator_rows(args.estimator_source)
    apply_field_qa(detector_rows, args.field_qa_substr)
    apply_field_qa(estimator_rows, args.field_qa_substr)
    splits = assign_global_splits(detector_rows, estimator_rows, args.val_frac, args.test_frac)
    for row in detector_rows + estimator_rows:
        row["split"] = row.get("split") or splits[str(row["source_group_id"])]

    registry_rows = build_registry_rows(detector_rows, estimator_rows)

    write_csv(
        args.output_dir / "global_split_registry.csv",
        registry_rows,
        [
            "source_group_id",
            "split",
            "split_reason",
            "source_dataset",
            "crab_id",
            "session_id",
            "collection_date",
            "is_in_situ",
            "color_state",
            "view",
            "species",
            "has_detector_label",
            "has_estimator_label",
            "has_negative_label",
            "negative_type",
            "notes",
        ],
    )
    write_csv(
        args.output_dir / "detector_v2_manifest.csv",
        detector_rows,
        [
            "image_path",
            "source_group_id",
            "split",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "species",
            "view",
            "color_state",
            "is_in_situ",
            "is_negative",
            "negative_type",
            "label_source",
            "label_confidence",
            "review_status",
            "notes",
        ],
    )
    write_csv(
        args.output_dir / "estimator_v2_manifest.csv",
        estimator_rows,
        [
            "image_path",
            "source_group_id",
            "split",
            "crab_id",
            "capture_date",
            "molt_date",
            "days_to_molt",
            "species",
            "view",
            "sex",
            "color_state",
            "is_in_situ",
            "crop_source",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "label_confidence",
            "notes",
        ],
    )
    print(f"Detector rows: {len(detector_rows)}")
    print(f"Estimator rows: {len(estimator_rows)}")
    print(f"Registry groups: {len(registry_rows)}")
    print(f"Wrote manifests under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
