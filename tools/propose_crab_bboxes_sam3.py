#!/usr/bin/env python3
"""
Propose crab bounding boxes with SAM3 text prompts and optional OpenCLIP ranking.

Primary use:
  /Users/gen/.venv/focal3.12/bin/python tools/propose_crab_bboxes_sam3.py \
    --input "data/raw/Green Crab AI 2026" \
    --output data/bootstrap_bboxes/blue_cooler_may29 \
    --max-images 0

Outputs:
- proposals.csv: one row per candidate bbox
- review_overlays/: original images with numbered candidate boxes
- crops/: candidate crop thumbnails for review
- contact_sheets/: quick visual review pages

The output is intentionally review-first: accept/reject decisions should be made
in proposals.csv or a copied review spreadsheet before exporting to YOLO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor

try:
    import open_clip
except Exception:  # pragma: no cover - open_clip is optional but recommended.
    open_clip = None


LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "crab",
    "a crab",
    "green crab",
    "crab in hand",
    "ventral crab underside",
    "dorsal crab shell",
    "side view of crab",
]
POSITIVE_CLIP_PROMPTS = [
    "a photo of a green crab",
    "a close-up photo of a crab in a hand",
    "a crab underside with legs and claws",
]
NEGATIVE_CLIP_PROMPTS = [
    "a photo of a glove without a crab",
    "a photo of a human hand",
    "a photo of a wooden table or wire mesh",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    image_path: str
    image_id: str
    candidate_id: str
    prompt: str
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int
    bbox_width: int
    bbox_height: int
    bbox_area_pct: float
    mask_area_pct: float
    aspect_ratio: float
    sam_rank: int
    clip_crab_score: Optional[float]
    clip_negative_score: Optional[float]
    candidate_score: float
    source_model: str
    overlay_path: str
    crop_path: str
    review_status: str = "new"
    review_notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory to scan.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for proposals.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for smoke tests; 0 means all.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS, help="SAM3 text prompts.")
    parser.add_argument("--min-area-pct", type=float, default=0.006, help="Reject masks smaller than this image fraction.")
    parser.add_argument("--max-area-pct", type=float, default=0.88, help="Reject masks larger than this image fraction.")
    parser.add_argument("--min-aspect", type=float, default=0.25, help="Minimum bbox width/height.")
    parser.add_argument("--max-aspect", type=float, default=4.0, help="Maximum bbox width/height.")
    parser.add_argument("--nms-iou", type=float, default=0.65, help="NMS IoU threshold across prompts.")
    parser.add_argument("--top-k", type=int, default=4, help="Max candidates to keep per image after NMS.")
    parser.add_argument(
        "--inference-max-side",
        type=int,
        default=0,
        help="Resize images so their longest side is at most this many pixels for SAM3 inference; 0 keeps original size.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Use raw SAM3 masks directly instead of processor.postprocess_outputs.",
    )
    parser.add_argument("--no-clip", action="store_true", help="Disable OpenCLIP crop ranking.")
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip images already listed in processed_images.csv.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write CSV checkpoints after this many images.")
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def stable_image_id(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def mask_to_bbox(mask: torch.Tensor, orig_h: int, orig_w: int) -> Optional[Tuple[int, int, int, int, float]]:
    if mask.dtype != torch.bool:
        mask = mask > 0
    if not mask.any():
        return None
    ys, xs = torch.nonzero(mask, as_tuple=True)
    y1 = int(math.floor(float(ys.min().item()) * orig_h / mask.shape[0]))
    y2 = int(math.ceil(float(ys.max().item() + 1) * orig_h / mask.shape[0]))
    x1 = int(math.floor(float(xs.min().item()) * orig_w / mask.shape[1]))
    x2 = int(math.ceil(float(xs.max().item() + 1) * orig_w / mask.shape[1]))
    mask_area_pct = float(mask.float().mean().item())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(orig_w, x2), min(orig_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, mask_area_pct


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_clip(device: torch.device):
    if open_clip is None:
        LOGGER.warning("open_clip is unavailable; candidate_score will use geometry/SAM only.")
        return None
    LOGGER.info("Loading OpenCLIP ViT-H/14 on %s", device)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pos_tokens = tokenizer(POSITIVE_CLIP_PROMPTS).to(device)
        neg_tokens = tokenizer(NEGATIVE_CLIP_PROMPTS).to(device)
        pos_text = model.encode_text(pos_tokens)
        neg_text = model.encode_text(neg_tokens)
        pos_text = pos_text / pos_text.norm(dim=-1, keepdim=True)
        neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)
    return model, preprocess, pos_text, neg_text


def score_crop(clip_state, crop: Image.Image, device: torch.device) -> Tuple[Optional[float], Optional[float], float]:
    if clip_state is None:
        return None, None, 0.0
    model, preprocess, pos_text, neg_text = clip_state
    with torch.no_grad():
        tensor = preprocess(crop).unsqueeze(0).to(device)
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        pos_score = float((feat @ pos_text.T).max().item())
        neg_score = float((feat @ neg_text.T).max().item())
    return pos_score, neg_score, pos_score - neg_score


def propose_for_prompt(
    img: Image.Image,
    prompt: str,
    model,
    processor,
    device: torch.device,
    filters: argparse.Namespace,
) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
    orig_w, orig_h = img.size
    session = processor.init_video_session(
        video=[img],
        inference_device=device,
        inference_state_device=device,
        processing_device=device,
        video_storage_device=device,
    )
    session = processor.add_text_prompt(session, prompt)
    with torch.no_grad():
        outputs = model(session, frame_idx=0)

    masks = None
    boxes = None
    scores = None
    if not filters.skip_postprocess:
        try:
            results = processor.postprocess_outputs(session, outputs)
            if results:
                masks = results.get("masks")
                boxes = results.get("boxes")
                scores = results.get("scores")
        except RuntimeError as exc:
            LOGGER.warning("SAM3 postprocess failed for prompt %r; using raw masks: %s", prompt, exc)

    if masks is None and getattr(outputs, "obj_id_to_mask", None):
        masks = []
        scores = []
        for object_id in outputs.object_ids:
            mask = outputs.obj_id_to_mask.get(object_id)
            if mask is None:
                continue
            masks.append(mask.squeeze().detach().cpu())
            score = outputs.obj_id_to_score.get(object_id, 0.0)
            scores.append(float(score))

    if masks is None or len(masks) == 0:
        return []
    proposals: List[Tuple[Tuple[int, int, int, int], float, int]] = []
    order = list(range(len(masks)))
    if scores is not None and len(scores) == len(order):
        order.sort(key=lambda idx: float(scores[idx].item() if hasattr(scores[idx], "item") else scores[idx]), reverse=True)
    for rank, idx in enumerate(order):
        mask = masks[idx].detach().cpu() if hasattr(masks[idx], "detach") else torch.as_tensor(masks[idx])
        if mask.ndim == 3:
            mask = mask.squeeze()
        bbox_from_mask = mask_to_bbox(mask, orig_h, orig_w)
        if bbox_from_mask is None:
            continue
        mask_x1, mask_y1, mask_x2, mask_y2, mask_area_pct = bbox_from_mask
        if boxes is not None:
            box = boxes[idx].detach().cpu().tolist() if hasattr(boxes[idx], "detach") else boxes[idx]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
        else:
            x1, y1, x2, y2 = mask_x1, mask_y1, mask_x2, mask_y2
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        bbox_area_pct = (w * h) / max(orig_w * orig_h, 1)
        aspect = w / max(h, 1)
        if bbox_area_pct < filters.min_area_pct or bbox_area_pct > filters.max_area_pct:
            continue
        if aspect < filters.min_aspect or aspect > filters.max_aspect:
            continue
        proposals.append(((x1, y1, x2, y2), mask_area_pct, rank))
    return proposals


def draw_overlay(image: Image.Image, candidates: Sequence[Candidate], out_path: Path) -> None:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    for idx, candidate in enumerate(candidates):
        color = colors[idx % len(colors)]
        box = [candidate.bbox_xmin, candidate.bbox_ymin, candidate.bbox_xmax, candidate.bbox_ymax]
        draw.rectangle(box, outline=color, width=5)
        text = f"{idx + 1}: {candidate.candidate_score:.2f} {candidate.prompt}"
        text_xy = (candidate.bbox_xmin + 4, max(0, candidate.bbox_ymin - 18))
        draw.rectangle([text_xy[0], text_xy[1], text_xy[0] + min(520, 7 * len(text)), text_xy[1] + 16], fill="black")
        draw.text(text_xy, text, fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.thumbnail((1800, 1800))
    overlay.save(out_path, quality=90)


def save_contact_sheets(overlay_paths: Sequence[Path], out_dir: Path, cols: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for path in overlay_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 420))
        thumbs.append((path.name, img.copy()))
    rows_per_sheet = 4
    per_sheet = cols * rows_per_sheet
    for sheet_idx in range(0, len(thumbs), per_sheet):
        batch = thumbs[sheet_idx : sheet_idx + per_sheet]
        rows = math.ceil(len(batch) / cols)
        sheet = Image.new("RGB", (cols * 440, rows * 470), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (name, img) in enumerate(batch):
            x = (idx % cols) * 440
            y = (idx // cols) * 470
            sheet.paste(img, (x + 10, y + 10))
            draw.text((x + 10, y + 435), name[:58], fill="black")
        sheet_num = sheet_idx // per_sheet + 1
        sheet.save(out_dir / f"contact_sheet_{sheet_num:03d}.jpg", quality=90)


def process_image(
    path: Path,
    model,
    processor,
    clip_state,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Candidate]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image_id = stable_image_id(path)
    orig_w, orig_h = image.size
    inference_image = image
    scale_x = 1.0
    scale_y = 1.0
    if args.inference_max_side and max(orig_w, orig_h) > args.inference_max_side:
        scale = args.inference_max_side / max(orig_w, orig_h)
        inf_w = max(1, int(round(orig_w * scale)))
        inf_h = max(1, int(round(orig_h * scale)))
        inference_image = image.resize((inf_w, inf_h), Image.Resampling.LANCZOS)
        scale_x = orig_w / inf_w
        scale_y = orig_h / inf_h
    raw_candidates = []
    for prompt in args.prompts:
        for bbox, mask_area_pct, rank in propose_for_prompt(inference_image, prompt, model, processor, device, args):
            x1, y1, x2, y2 = bbox
            scaled_bbox = (
                max(0, min(orig_w, int(round(x1 * scale_x)))),
                max(0, min(orig_h, int(round(y1 * scale_y)))),
                max(0, min(orig_w, int(round(x2 * scale_x)))),
                max(0, min(orig_h, int(round(y2 * scale_y)))),
            )
            if scaled_bbox[2] <= scaled_bbox[0] or scaled_bbox[3] <= scaled_bbox[1]:
                continue
            raw_candidates.append((prompt, scaled_bbox, mask_area_pct, rank))

    scored = []
    for prompt, bbox, mask_area_pct, rank in raw_candidates:
        x1, y1, x2, y2 = bbox
        crop = image.crop((x1, y1, x2, y2))
        clip_pos, clip_neg, clip_delta = score_crop(clip_state, crop, device)
        area_pct = ((x2 - x1) * (y2 - y1)) / max(orig_w * orig_h, 1)
        geometry_score = min(area_pct, 0.45) - abs(((x2 - x1) / max(y2 - y1, 1)) - 1.2) * 0.02
        candidate_score = clip_delta + geometry_score - rank * 0.02
        scored.append((candidate_score, prompt, bbox, mask_area_pct, rank, clip_pos, clip_neg))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = []
    for item in scored:
        bbox = item[2]
        if all(bbox_iou(bbox, previous[2]) < args.nms_iou for previous in kept):
            kept.append(item)
        if len(kept) >= args.top_k:
            break

    candidates: List[Candidate] = []
    overlay_rel = Path("review_overlays") / f"{image_id}_overlay.jpg"
    for idx, (candidate_score, prompt, bbox, mask_area_pct, rank, clip_pos, clip_neg) in enumerate(kept, start=1):
        x1, y1, x2, y2 = bbox
        crop_rel = Path("crops") / f"{image_id}_candidate_{idx:02d}.jpg"
        crop_path = args.output / crop_rel
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop = image.crop((x1, y1, x2, y2))
        crop.thumbnail((512, 512))
        crop.save(crop_path, quality=90)
        w, h = x2 - x1, y2 - y1
        candidates.append(
            Candidate(
                image_path=str(path),
                image_id=image_id,
                candidate_id=f"{image_id}_candidate_{idx:02d}",
                prompt=prompt,
                bbox_xmin=x1,
                bbox_ymin=y1,
                bbox_xmax=x2,
                bbox_ymax=y2,
                bbox_width=w,
                bbox_height=h,
                bbox_area_pct=round((w * h) / max(orig_w * orig_h, 1), 6),
                mask_area_pct=round(mask_area_pct, 6),
                aspect_ratio=round(w / max(h, 1), 4),
                sam_rank=rank,
                clip_crab_score=round(clip_pos, 6) if clip_pos is not None else None,
                clip_negative_score=round(clip_neg, 6) if clip_neg is not None else None,
                candidate_score=round(candidate_score, 6),
                source_model="facebook/sam3 + OpenCLIP ViT-H/14" if clip_state else "facebook/sam3",
                overlay_path=str(overlay_rel),
                crop_path=str(crop_rel),
            )
        )
    draw_overlay(image, candidates, args.output / overlay_rel)
    return candidates


def write_csv(rows: Sequence[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(Candidate.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_existing_candidates(out_path: Path) -> List[Candidate]:
    if not out_path.exists():
        return []
    rows: List[Candidate] = []
    with out_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in [
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "bbox_width",
                "bbox_height",
                "sam_rank",
            ]:
                row[key] = int(row[key])
            for key in [
                "bbox_area_pct",
                "mask_area_pct",
                "aspect_ratio",
                "candidate_score",
            ]:
                row[key] = float(row[key])
            for key in ["clip_crab_score", "clip_negative_score"]:
                row[key] = float(row[key]) if row[key] not in ("", None) else None
            rows.append(Candidate(**row))
    return rows


def read_processed_images(out_path: Path) -> List[Dict[str, str]]:
    if not out_path.exists():
        return []
    with out_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_processed_images(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "image_id", "candidate_count", "status"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.input)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    proposals_path = args.output / "proposals.csv"
    processed_path = args.output / "processed_images.csv"
    all_candidates: List[Candidate] = read_existing_candidates(proposals_path) if args.resume else []
    processed_rows = read_processed_images(processed_path) if args.resume else []
    processed_paths = {row["image_path"] for row in processed_rows}
    if args.resume:
        images = [path for path in images if str(path) not in processed_paths]
        LOGGER.info("Resuming with %d existing candidates and %d remaining images", len(all_candidates), len(images))

    device = choose_device(args.device)
    LOGGER.info("Processing %d images on %s", len(images), device)
    LOGGER.info("Loading SAM3")
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device)
    model.eval()
    clip_state = None if args.no_clip else load_clip(device)

    overlay_paths: List[Path] = [
        args.output / row.overlay_path for row in all_candidates if row.overlay_path
    ]
    for image_num, path in enumerate(tqdm(images, desc="SAM3 bbox proposals"), start=1):
        image_id = stable_image_id(path)
        try:
            candidates = process_image(path, model, processor, clip_state, device, args)
            all_candidates.extend(candidates)
            overlay_paths.append(args.output / "review_overlays" / f"{stable_image_id(path)}_overlay.jpg")
            processed_rows.append(
                {
                    "image_path": str(path),
                    "image_id": image_id,
                    "candidate_count": str(len(candidates)),
                    "status": "ok",
                }
            )
        except Exception:
            LOGGER.exception("Failed to process %s", path)
            processed_rows.append(
                {
                    "image_path": str(path),
                    "image_id": image_id,
                    "candidate_count": "0",
                    "status": "failed",
                }
            )
        if image_num % args.checkpoint_every == 0:
            write_csv(all_candidates, proposals_path)
            write_processed_images(processed_rows, processed_path)

    write_csv(all_candidates, proposals_path)
    write_processed_images(processed_rows, processed_path)
    save_contact_sheets(overlay_paths, args.output / "contact_sheets", args.contact_sheet_cols)
    LOGGER.info("Wrote %d candidates to %s", len(all_candidates), args.output / "proposals.csv")
    LOGGER.info("Review overlays: %s", args.output / "review_overlays")
    LOGGER.info("Contact sheets: %s", args.output / "contact_sheets")


if __name__ == "__main__":
    main()
