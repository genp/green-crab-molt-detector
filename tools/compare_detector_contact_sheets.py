#!/usr/bin/env python3
"""
Compare two YOLO detectors on a dogfood image set and emit contact sheets.

For each source image this script:
- runs both detectors on the original image
- runs both detectors on a downscaled + JPEG re-encoded version
- saves side-by-side comparison panels
- builds contact sheets from those panels
- writes a CSV with every detected bbox and timing measurement
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetectorSpec:
    name: str
    path: Path
    model: YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--baseline-model", type=Path, required=True, help="Baseline detector weights.")
    parser.add_argument("--candidate-model", type=Path, required=True, help="Candidate detector weights.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--max-det", type=int, default=10, help="Max detections per image.")
    parser.add_argument("--max-dimension", type=int, default=416, help="Downscale max dimension for stream simulation.")
    parser.add_argument("--jpeg-quality", type=int, default=65, help="JPEG quality for stream simulation.")
    parser.add_argument("--cols", type=int, default=4, help="Contact sheet columns.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for smoke tests; 0 means all.")
    return parser.parse_args()


def collect_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def stable_image_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def downscale_and_reencode(image: Image.Image, max_dimension: int, jpeg_quality: int) -> Image.Image:
    working = image.copy().convert("RGB")
    max_side = max(working.size)
    if max_side > max_dimension:
        scale = max_dimension / max_side
        new_size = (max(1, round(working.width * scale)), max(1, round(working.height * scale)))
        working = working.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    working.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    buf.seek(0)
    with Image.open(buf) as reopened:
        return reopened.convert("RGB")


def run_detector(spec: DetectorSpec, image: Image.Image, imgsz: int, conf: float, max_det: int) -> Tuple[List[Dict[str, float]], float]:
    started = time.perf_counter()
    results = spec.model.predict(image, imgsz=imgsz, conf=conf, max_det=max_det, verbose=False)
    detections: List[Dict[str, float]] = []
    if results and getattr(results[0], "boxes", None) is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(
                {
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                    "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
                    "class": int(box.cls[0]) if box.cls is not None else -1,
                }
            )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return detections, elapsed_ms


def draw_overlay(image: Image.Image, detections: Sequence[Dict[str, float]], title: str, subtitle: str) -> Image.Image:
    bar_height = 40
    canvas = Image.new("RGB", (image.width, image.height + bar_height), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, canvas.width, bar_height], fill=(18, 18, 18))
    draw.text((10, 10), title, fill="white", font=font)
    draw.text((10 + min(240, canvas.width // 2), 10), subtitle, fill="white", font=font)

    for idx, det in enumerate(detections, start=1):
        x1 = det["xmin"]
        y1 = det["ymin"] + bar_height
        x2 = det["xmax"]
        y2 = det["ymax"] + bar_height
        conf = det.get("confidence", 0.0)
        label = f"{idx}:{conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=4)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = text_bbox[2] - text_bbox[0] + 8
        label_h = text_bbox[3] - text_bbox[1] + 6
        label_y = max(0, y1 - label_h - 2)
        draw.rectangle([x1, label_y, x1 + label_w, label_y + label_h], fill=(0, 200, 255))
        draw.text((x1 + 4, label_y + 2), label, fill="black", font=font)

    return canvas


def make_panel(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    gap = 12
    caption_h = 34
    width = left.width + right.width + gap * 3
    height = max(left.height, right.height) + caption_h + gap * 2
    panel = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((gap, 10), caption, fill=(20, 20, 20), font=font)
    panel.paste(left, (gap, caption_h))
    panel.paste(right, (left.width + gap * 2, caption_h))
    return panel


def save_contact_sheets(paths: Sequence[Path], out_dir: Path, cols: int) -> None:
    if not paths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = [Image.open(path).convert("RGB") for path in paths]
    try:
        for start in range(0, len(thumbs), cols * 3):
            batch = thumbs[start : start + cols * 3]
            widths = [img.width for img in batch]
            heights = [img.height for img in batch]
            cell_w = max(widths)
            cell_h = max(heights)
            sheet_cols = min(cols, len(batch))
            sheet_rows = (len(batch) + sheet_cols - 1) // sheet_cols
            sheet = Image.new("RGB", (sheet_cols * cell_w, sheet_rows * cell_h), (255, 255, 255))
            for index, img in enumerate(batch):
                row = index // sheet_cols
                col = index % sheet_cols
                x = col * cell_w + (cell_w - img.width) // 2
                y = row * cell_h + (cell_h - img.height) // 2
                sheet.paste(img, (x, y))
            sheet.save(out_dir / f"contact_sheet_{start // (cols * 3) + 1:03d}.jpg", quality=90)
    finally:
        for img in thumbs:
            img.close()


def bbox_row(
    image_name: str,
    mode: str,
    detector: str,
    elapsed_ms: float,
    detections: Sequence[Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, det in enumerate(detections, start=1):
        rows.append(
            {
                "image": image_name,
                "mode": mode,
                "detector": detector,
                "bbox_index": index,
                "xmin": round(det["xmin"], 2),
                "ymin": round(det["ymin"], 2),
                "xmax": round(det["xmax"], 2),
                "ymax": round(det["ymax"], 2),
                "confidence": round(det.get("confidence", 0.0), 4),
                "class_id": det.get("class", -1),
                "elapsed_ms": round(elapsed_ms, 1),
                "bbox_count": len(detections),
            }
        )
    if not rows:
        rows.append(
            {
                "image": image_name,
                "mode": mode,
                "detector": detector,
                "bbox_index": 0,
                "xmin": "",
                "ymin": "",
                "xmax": "",
                "ymax": "",
                "confidence": "",
                "class_id": "",
                "elapsed_ms": round(elapsed_ms, 1),
                "bbox_count": 0,
            }
        )
    return rows


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    images = collect_images(args.input)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    baseline = DetectorSpec("yolov8n", args.baseline_model, YOLO(str(args.baseline_model)))
    candidate = DetectorSpec("bootstrapv1", args.candidate_model, YOLO(str(args.candidate_model)))

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    original_dir = root / "original"
    downscaled_dir = root / "downscaled_416_q65"
    original_dir.mkdir(parents=True, exist_ok=True)
    downscaled_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    original_panels: List[Path] = []
    downscaled_panels: List[Path] = []

    for image_path in images:
        image = load_image(image_path)
        stream_image = downscale_and_reencode(image, args.max_dimension, args.jpeg_quality)
        image_id = stable_image_id(image_path)

        original_outputs = []
        downscaled_outputs = []
        for mode_name, mode_image, mode_dir, panel_bucket in (
            ("original", image, original_dir, original_panels),
            ("downscaled_416_q65", stream_image, downscaled_dir, downscaled_panels),
        ):
            base_dets, base_ms = run_detector(baseline, mode_image, args.imgsz, args.conf, args.max_det)
            cand_dets, cand_ms = run_detector(candidate, mode_image, args.imgsz, args.conf, args.max_det)
            all_rows.extend(bbox_row(image_path.name, mode_name, baseline.name, base_ms, base_dets))
            all_rows.extend(bbox_row(image_path.name, mode_name, candidate.name, cand_ms, cand_dets))

            base_overlay = draw_overlay(
                mode_image,
                base_dets,
                f"{baseline.name} | {mode_name}",
                f"{len(base_dets)} boxes | {base_ms:.0f} ms",
            )
            cand_overlay = draw_overlay(
                mode_image,
                cand_dets,
                f"{candidate.name} | {mode_name}",
                f"{len(cand_dets)} boxes | {cand_ms:.0f} ms",
            )

            panel = make_panel(
                base_overlay,
                cand_overlay,
                f"{image_path.name} | {mode_name} | baseline vs candidate",
            )
            panel_path = mode_dir / f"{image_id}_{mode_name}_comparison.jpg"
            panel.save(panel_path, quality=92)
            panel_bucket.append(panel_path)

    save_contact_sheets(original_panels, original_dir / "contact_sheets", args.cols)
    save_contact_sheets(downscaled_panels, downscaled_dir / "contact_sheets", args.cols)
    write_csv(all_rows, root / "bbox_summary.csv")

    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in all_rows:
        key = (str(row["mode"]), str(row["detector"]))
        bucket = summary.setdefault(key, {"count": 0.0, "elapsed_sum": 0.0, "bbox_sum": 0.0})
        bucket["count"] += 1
        bucket["elapsed_sum"] += float(row["elapsed_ms"])
        bucket["bbox_sum"] += float(row["bbox_count"])

    print(f"Images processed: {len(images)}")
    print(f"Comparison panels: {len(original_panels) + len(downscaled_panels)}")
    print(f"Contact sheets: {original_dir / 'contact_sheets'}")
    print(f"Contact sheets: {downscaled_dir / 'contact_sheets'}")
    print(f"BBox summary: {root / 'bbox_summary.csv'}")
    for (mode, detector), stats in sorted(summary.items()):
        count = stats["count"] or 1.0
        print(
            f"{mode} | {detector}: "
            f"avg_elapsed_ms={stats['elapsed_sum'] / count:.1f}, "
            f"avg_bbox_count={stats['bbox_sum'] / count:.2f}"
        )


if __name__ == "__main__":
    main()
