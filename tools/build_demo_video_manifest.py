"""Build cached demo video assets for the MoltMeter table demo.

The script intentionally uses the app's existing prediction code so the demo
manifest matches what the web app would return for an uploaded image. Videos
are downscaled with OpenCV when possible and each clip gets one cached
representative prediction.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import cv2
from fastapi import UploadFile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = Path.home() / "Downloads"
OUTPUT_DIR = REPO_ROOT / "static" / "demo_videos"
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
MAX_CLIPS = 11
MAX_SIDE = 960
TARGET_FPS = 24.0
JPEG_QUALITY = 88


def newest_videos() -> list[Path]:
    videos = [
        path
        for path in DOWNLOADS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.stat().st_mtime, reverse=True)[:MAX_CLIPS]


def scaled_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(width, height))
    out_w = max(2, int(round(width * scale)))
    out_h = max(2, int(round(height * scale)))
    return out_w - (out_w % 2), out_h - (out_h % 2)


def write_jpeg(path: Path, frame: Any) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def transcode_video(source: Path, destination: Path, thumb_path: Path, frame_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if Path(ffmpeg).exists():
        temp_output = destination.with_suffix(".tmp.mp4")
        temp_output.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-an",
                    "-vf",
                    "scale='min(960,iw)':-2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temp_output),
                ],
                check=True,
            )
            temp_output.replace(destination)
        except (OSError, subprocess.CalledProcessError):
            temp_output.unlink(missing_ok=True)
        else:
            return extract_representative_frames(destination, thumb_path, frame_path)

    return transcode_video_with_opencv(source, destination, thumb_path, frame_path)


def extract_representative_frames(source: Path, thumb_path: Path, frame_path: Path) -> bool:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False

    out_w, out_h = scaled_size(width, height)
    fps = min(TARGET_FPS, source_fps) if source_fps > 0 else TARGET_FPS
    sample_index = max(0, total_frames // 3) if total_frames else 0
    frame_index = 0
    representative = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (out_w, out_h) != (width, height):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if representative is None or frame_index == sample_index:
            representative = frame.copy()
        frame_index += 1

    cap.release()

    if representative is None:
        return False

    write_jpeg(thumb_path, representative)
    write_jpeg(frame_path, representative)
    return True


def transcode_video_with_opencv(source: Path, destination: Path, thumb_path: Path, frame_path: Path) -> bool:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False

    out_w, out_h = scaled_size(width, height)
    fps = min(TARGET_FPS, source_fps) if source_fps > 0 else TARGET_FPS
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        return False

    sample_index = max(0, total_frames // 3) if total_frames else 0
    frame_index = 0
    representative = None
    wrote_any = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (out_w, out_h) != (width, height):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if representative is None or frame_index == sample_index:
            representative = frame.copy()
        writer.write(frame)
        wrote_any = True
        frame_index += 1

    writer.release()
    cap.release()

    if not wrote_any or representative is None:
        destination.unlink(missing_ok=True)
        return False

    write_jpeg(thumb_path, representative)
    write_jpeg(frame_path, representative)
    return True


async def predict_frame(frame_path: Path) -> dict[str, Any]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".tmp" / "ultralytics"))
    sys.path.insert(0, str(REPO_ROOT))
    import app_fastapi  # noqa: PLC0415

    if not app_fastapi.models_ready.is_set():
        app_fastapi.load_models(load_detector=True)
        app_fastapi.models_ready.set()

    data = frame_path.read_bytes()
    spooled = SpooledTemporaryFile()
    spooled.write(data)
    spooled.seek(0)
    upload = UploadFile(file=spooled, filename=frame_path.name)
    result = await app_fastapi.predict_image(upload, stream_mode=False, include_thumbnail=True)
    return dict(result)


def fallback_copy(source: Path, destination: Path, thumb_path: Path, frame_path: Path) -> None:
    shutil.copy2(source, destination)
    cap = cv2.VideoCapture(str(destination))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read video frame from {source}")
    write_jpeg(thumb_path, frame)
    write_jpeg(frame_path, frame)


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".tmp" / "ultralytics").mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    for index, source in enumerate(newest_videos(), start=1):
        clip_id = f"demo_{index:02d}"
        video_path = OUTPUT_DIR / f"{clip_id}.mp4"
        thumb_path = OUTPUT_DIR / f"{clip_id}.jpg"
        frame_path = OUTPUT_DIR / f"{clip_id}_frame.jpg"

        converted = transcode_video(source, video_path, thumb_path, frame_path)
        converter = "opencv" if converted else "copy"
        if not converted:
            fallback_copy(source, video_path, thumb_path, frame_path)

        prediction = await predict_frame(frame_path)
        prediction.pop("thumbnail", None)
        clips.append(
            {
                "id": clip_id,
                "title": source.stem,
                "source": str(source),
                "video": f"/static/demo_videos/{video_path.name}",
                "thumbnail": f"/static/demo_videos/{thumb_path.name}",
                "frame": f"/static/demo_videos/{frame_path.name}",
                "converted": converted,
                "converter": converter,
                "prediction": prediction,
            }
        )
        print(f"{clip_id}: {source.name} -> {prediction.get('phase')} {prediction.get('days_until_molt')}")

    manifest = {
        "generated_by": "tools/build_demo_video_manifest.py",
        "source_dir": str(DOWNLOADS_DIR),
        "clips": clips,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
