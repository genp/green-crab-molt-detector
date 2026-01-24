#!/usr/bin/env python3
"""
Build a pseudo-labeled YOLO dataset from close-crop crab images.

Assumes each image is a tight crop of a crab and assigns a full-frame bbox.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from shutil import copy2


def collect_images(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def write_label(path: Path) -> None:
    # YOLO format: class_id cx cy w h (normalized)
    path.write_text("0 0.5 0.5 1.0 1.0\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "data" / "sam3_orientation"
    out_root = repo_root / "data" / "yolo_pseudo"
    images = collect_images(src_root)
    if not images:
        raise SystemExit(f"No images found under {src_root}")

    rng = Random(42)
    rng.shuffle(images)
    total = len(images)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in (
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ):
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in split_images:
            dest_img = img_out / img.name
            dest_lbl = lbl_out / (img.stem + ".txt")
            copy2(img, dest_img)
            write_label(dest_lbl)

    yaml_path = repo_root / "data" / "yolo_pseudo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: crab",
                "",
            ]
        )
    )

    print(f"Total images: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")
    print(f"Test: {len(test_images)}")
    print(f"Wrote dataset to {out_root}")
    print(f"Wrote config to {yaml_path}")


if __name__ == "__main__":
    main()
