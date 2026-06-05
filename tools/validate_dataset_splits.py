#!/usr/bin/env python3
"""
Validate that detector, estimator, and QA manifests use one global split registry.

Example:
  python3 tools/validate_dataset_splits.py \
    --registry data/processed/global_split_registry.csv \
    --manifest data/processed/detector_v2_manifest.csv detector \
    --manifest data/processed/estimator_v2_manifest.csv estimator \
    --manifest data/processed/field_qa_manifest.csv field_qa
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALLOWED_SPLITS = {"train", "val", "test", "field_qa_holdout"}
REQUIRED_REGISTRY_COLUMNS = {"source_group_id", "split"}
REQUIRED_MANIFEST_COLUMNS = {"source_group_id", "split"}
TRUTHY = {"1", "true", "yes", "y"}


Row = Dict[str, str]
Manifest = Tuple[Path, str, List[Row]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="Global split registry CSV.")
    parser.add_argument(
        "--manifest",
        nargs=2,
        action="append",
        metavar=("CSV", "ROLE"),
        default=[],
        help="Manifest CSV plus role: detector, estimator, field_qa, species, view, or temporal.",
    )
    return parser.parse_args()


def read_csv(path: Path, required_columns: Iterable[str]) -> List[Row]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = set(required_columns) - fieldnames
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            row["source_group_id"] = str(row.get("source_group_id", "")).strip()
            row["split"] = str(row.get("split", "")).strip()
            rows.append(row)
    return rows


def split_counts(rows: Iterable[Row]) -> Counter[str]:
    return Counter(row.get("split", "") for row in rows)


def validate_registry(registry: List[Row], errors: List[str]) -> Dict[str, str]:
    bad_splits = sorted({row["split"] for row in registry} - ALLOWED_SPLITS)
    if bad_splits:
        errors.append(f"Registry has invalid split values: {bad_splits}")

    splits_by_group: Dict[str, set[str]] = defaultdict(set)
    for row in registry:
        splits_by_group[row["source_group_id"]].add(row["split"])

    conflicting = sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)
    if conflicting:
        errors.append(
            "Registry source_group_id appears in multiple splits: " + ", ".join(conflicting[:20])
        )

    registry_splits = {}
    for row in registry:
        registry_splits.setdefault(row["source_group_id"], row["split"])
    return registry_splits


def validate_manifest_against_registry(
    *,
    manifest_path: Path,
    role: str,
    manifest: List[Row],
    registry_splits: Dict[str, str],
    errors: List[str],
) -> None:
    bad_splits = sorted({row["split"] for row in manifest} - ALLOWED_SPLITS)
    if bad_splits:
        errors.append(f"{manifest_path} has invalid split values: {bad_splits}")

    missing_groups = sorted({row["source_group_id"] for row in manifest} - set(registry_splits))
    if missing_groups:
        errors.append(
            f"{manifest_path} has source_group_id values missing from registry: "
            + ", ".join(missing_groups[:20])
        )

    splits_by_group: Dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        group_id = row["source_group_id"]
        split = row["split"]
        splits_by_group[group_id].add(split)
        registry_split = registry_splits.get(group_id)
        if registry_split is not None and split != registry_split:
            errors.append(
                f"{manifest_path} split mismatch for {group_id}: "
                f"manifest={split}, registry={registry_split}"
            )

        is_training_row = row.get("is_training_row", "true").strip().lower() in TRUTHY
        if role != "field_qa" and split == "field_qa_holdout" and is_training_row:
            errors.append(f"{manifest_path} has training rows assigned to field_qa_holdout")

    conflicting = sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)
    if conflicting:
        errors.append(
            f"{manifest_path} source_group_id appears in multiple splits: "
            + ", ".join(conflicting[:20])
        )


def validate_cross_model_roles(manifests: List[Manifest], errors: List[str]) -> None:
    groups_by_role_split: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    image_splits: Dict[str, set[str]] = defaultdict(set)

    for _, role, rows in manifests:
        for row in rows:
            groups_by_role_split[(role, row["split"])].add(row["source_group_id"])
            image_path = row.get("image_path", "").strip()
            if image_path:
                image_splits[image_path].add(row["split"])

    for image_path, splits in image_splits.items():
        if len(splits) > 1:
            errors.append(f"Image path appears in conflicting splits: {image_path} -> {sorted(splits)}")

    detector_train = groups_by_role_split.get(("detector", "train"), set())
    estimator_eval = groups_by_role_split.get(("estimator", "val"), set()) | groups_by_role_split.get(
        ("estimator", "test"), set()
    )
    overlap = sorted(detector_train & estimator_eval)
    if overlap:
        errors.append("Detector train groups appear in estimator val/test: " + ", ".join(overlap[:20]))

    estimator_train = groups_by_role_split.get(("estimator", "train"), set())
    detector_eval = groups_by_role_split.get(("detector", "val"), set()) | groups_by_role_split.get(
        ("detector", "test"), set()
    )
    overlap = sorted(estimator_train & detector_eval)
    if overlap:
        errors.append("Estimator train groups appear in detector val/test: " + ", ".join(overlap[:20]))


def subgroup_counts(rows: List[Row], column: str, value: str) -> Dict[str, int]:
    counts = Counter(
        row.get("split", "")
        for row in rows
        if row.get(column, "").strip().lower() == value
    )
    return dict(sorted(counts.items()))


def in_situ_counts(rows: List[Row]) -> Dict[str, int]:
    counts = Counter(
        row.get("split", "")
        for row in rows
        if row.get("is_in_situ", "").strip().lower() in TRUTHY
    )
    return dict(sorted(counts.items()))


def print_subgroup_report(registry: List[Row], manifests: List[Manifest]) -> None:
    print("Registry split counts:")
    for split, count in sorted(split_counts(registry).items()):
        print(f"  {split}: {count}")
    print()

    def report(rows: List[Row], label: str) -> None:
        print(label)
        checks = [
            ("color_state", "red_green_crab"),
            ("view", "side"),
            ("negative_type", "human"),
            ("negative_type", "glove"),
            ("negative_type", "equipment"),
        ]
        for column, value in checks:
            counts = subgroup_counts(rows, column, value)
            if counts:
                print(f"  {column}={value}: {counts}")
        counts = in_situ_counts(rows)
        if counts:
            print(f"  is_in_situ=true: {counts}")
        print()

    report(registry, "Registry subgroup counts:")
    for path, role, rows in manifests:
        report(rows, f"{role} subgroup counts ({path}):")


def main() -> int:
    args = parse_args()
    errors: List[str] = []

    try:
        registry = read_csv(args.registry, REQUIRED_REGISTRY_COLUMNS)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    registry_splits = validate_registry(registry, errors)

    manifests: List[Manifest] = []
    for manifest_arg, role in args.manifest:
        path = Path(manifest_arg)
        try:
            manifest = read_csv(path, REQUIRED_MANIFEST_COLUMNS)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{exc}")
            continue
        role = role.strip().lower()
        manifests.append((path, role, manifest))
        validate_manifest_against_registry(
            manifest_path=path,
            role=role,
            manifest=manifest,
            registry_splits=registry_splits,
            errors=errors,
        )

    validate_cross_model_roles(manifests, errors)
    print_subgroup_report(registry, manifests)

    if errors:
        print("Split validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Split validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
