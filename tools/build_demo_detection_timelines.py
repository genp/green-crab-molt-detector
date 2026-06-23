"""Precompute per-video demo detections at the stream detector cadence."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import cv2
from fastapi import UploadFile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "static" / "demo_videos"
INTERVAL_SECONDS = 0.2
MAX_SIDE = 416
JPEG_QUALITY = 82


def compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "success",
        "days_until_molt",
        "raw_days_until_molt",
        "estimate_range_days",
        "estimate_smoothed",
        "estimated_molt_event_date",
        "phase",
        "color",
        "recommendation",
        "harvest_ready",
        "confidence",
        "app_estimate_input",
        "whole_image_fallback_used",
        "feature_type",
        "model_display_name",
        "bbox_count",
        "crop_used",
        "stream_mode",
        "bbox_cached",
        "bbox_stale",
        "bbox_cleared_reason",
        "last_detection_age_ms",
        "estimate_cached",
        "estimate_stale",
        "primary_bbox",
        "bboxes",
        "image_width",
        "image_height",
    ]
    return {key: prediction.get(key) for key in keys if key in prediction}


def frame_to_jpeg_bytes(frame: Any) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    scale = min(1.0, MAX_SIDE / max(image.width, image.height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


async def predict_frame(app_fastapi: Any, frame: Any, filename: str) -> dict[str, Any]:
    data = frame_to_jpeg_bytes(frame)
    spooled = SpooledTemporaryFile()
    spooled.write(data)
    spooled.seek(0)
    upload = UploadFile(file=spooled, filename=filename)
    prediction = await app_fastapi.predict_image(
        upload,
        stream_mode=True,
        include_thumbnail=False,
        detection_imgsz=app_fastapi.STREAM_YOLO_IMGSZ,
    )
    return compact_prediction(dict(prediction))


async def build_timeline(video_path: Path, output_path: Path, app_fastapi: Any) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    sample_times = []
    t = 0.0
    while t < duration:
        sample_times.append(round(t, 3))
        t += INTERVAL_SECONDS

    app_fastapi.clear_stream_cache()
    samples = []
    for index, sample_time in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        prediction = await predict_frame(app_fastapi, frame, f"{video_path.stem}_{index:04d}.jpg")
        samples.append({"t": sample_time, "prediction": prediction})
        if index % 10 == 0:
            print(f"{video_path.name}: {index + 1}/{len(sample_times)}")
    cap.release()

    output_path.write_text(
        json.dumps(
            {
                "video": f"/static/demo_videos/{video_path.name}",
                "interval_seconds": INTERVAL_SECONDS,
                "duration_seconds": duration,
                "samples": samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".tmp" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
    sys.path.insert(0, str(REPO_ROOT))
    import app_fastapi  # noqa: PLC0415

    if not app_fastapi.models_ready.is_set():
        app_fastapi.load_models(load_detector=True)
        app_fastapi.models_ready.set()

    manifest_path = DEMO_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for clip in manifest.get("clips", []):
        video_path = REPO_ROOT / clip["video"].lstrip("/")
        timeline_path = DEMO_DIR / f"{Path(video_path).stem}_timeline.json"
        await build_timeline(video_path, timeline_path, app_fastapi)
        clip["timeline"] = f"/static/demo_videos/{timeline_path.name}"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
