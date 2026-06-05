"""
FastAPI application for green crab molt phase detection with streaming support.

Features:
- Single image prediction (`/predict`)
- Streaming frames endpoint (`/predict_stream`) returning thumbnail overlays and optional bboxes
- Health check (`/health`)

Detection bboxes are returned when a YOLO detector is configured (via env var
YOLO_MODEL_PATH); otherwise the list is empty.
"""

import asyncio
import base64
import json
import logging
import os
import re
import sys
import shutil
import threading
import time
import uuid
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

# Patch for torch/transformers compatibility issue (mirrors flask app)
import torch

if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32
if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from feature_extractor import GeneralCrustaceanFeatureExtractor  # noqa: E402
from model import MoltPhaseRegressor  # noqa: E402

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # noqa: N816

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Green Crab Molt Detector", version="1.0.0")

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=*, microphone=(), geolocation=()"
    # Add Content Security Policy for additional trust
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https:; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "font-src 'self' https://cdn.jsdelivr.net;"
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_PATH = Path(__file__).parent
MODELS_DIR = BASE_PATH / "models"
UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
DEFAULT_YOLO_MODEL_PATH = MODELS_DIR / "fathomnet_mvp_yolov8_1280_20240914.pt"
DEFAULT_LIGHT_YOLO_MODEL_PATH = (
    MODELS_DIR / "yolov8n.pt" if (MODELS_DIR / "yolov8n.pt").exists() else BASE_PATH / "yolov8n.pt"
)
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "cpu").lower()
DETECTION_ENABLED = os.getenv("DETECTION_ENABLED", "true").lower() == "true"
DETECTION_CROP_ENABLED = os.getenv("DETECTION_CROP_ENABLED", "true").lower() == "true"
YOLO_CONF_MIN = float(os.getenv("YOLO_CONF_MIN", "0.35"))
YOLO_MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "10"))
YOLO_MIN_AREA_PCT = float(os.getenv("YOLO_MIN_AREA_PCT", "0.01"))
YOLO_MAX_AREA_PCT = float(os.getenv("YOLO_MAX_AREA_PCT", "0.8"))
YOLO_MIN_ASPECT = float(os.getenv("YOLO_MIN_ASPECT", "0.5"))
YOLO_MAX_ASPECT = float(os.getenv("YOLO_MAX_ASPECT", "2.0"))
YOLO_NMS_IOU = float(os.getenv("YOLO_NMS_IOU", "0.45"))
STREAM_YOLO_IMGSZ = int(os.getenv("STREAM_YOLO_IMGSZ", "416"))
STREAM_ESTIMATE_EVERY_MS = int(os.getenv("STREAM_ESTIMATE_EVERY_MS", "1000"))
STREAM_CACHE_TTL_MS = int(os.getenv("STREAM_CACHE_TTL_MS", "800"))
STREAM_BBOX_REUSE_IOU = float(os.getenv("STREAM_BBOX_REUSE_IOU", "0.65"))
STREAM_SKIP_THUMBNAIL = os.getenv("STREAM_SKIP_THUMBNAIL", "true").lower() == "true"
DEBUG_EXPORTS_DIR = BASE_PATH / "data" / "debug_exports"
DEBUG_SESSION_DIR = DEBUG_EXPORTS_DIR / "current"
YOLO_MODEL_PATH = (
    Path(os.getenv("YOLO_MODEL_PATH"))
    if os.getenv("YOLO_MODEL_PATH")
    else (
        DEFAULT_YOLO_MODEL_PATH
        if INFERENCE_MODE == "cpu" and DEFAULT_YOLO_MODEL_PATH.exists()
        else (
            DEFAULT_LIGHT_YOLO_MODEL_PATH
            if INFERENCE_MODE == "cpu" and DEFAULT_LIGHT_YOLO_MODEL_PATH.exists()
            else (DEFAULT_YOLO_MODEL_PATH if DEFAULT_YOLO_MODEL_PATH.exists() else None)
        )
    )
)
TEMPLATE_PATH = BASE_PATH / "templates" / "index.html"

# Optional static mount (for any future assets)
if (BASE_PATH / "static").exists():
    app.mount("/static", StaticFiles(directory=BASE_PATH / "static"), name="static")

# Globals
feature_extractor: Optional[GeneralCrustaceanFeatureExtractor] = None
regressor: Optional[MoltPhaseRegressor] = None
feature_type: Optional[str] = None
regressor_model_path: Optional[Path] = None
detector_model_path: Optional[Path] = None
yolo_detector = None
yolo_class_names: Optional[Dict[int, str]] = None
inference_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCES", "2")))
models_ready = threading.Event()
model_load_lock = threading.Lock()
stream_cache_lock = threading.Lock()
stream_cache: Dict[str, object] = {}
debug_session_lock = threading.Lock()
debug_session_state: Dict[str, Any] = {
    "active": False,
    "session_id": None,
    "run_name": None,
    "location_name": None,
    "started_at_utc": None,
    "ended_at_utc": None,
    "capture_count": 0,
}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in UPLOAD_EXTENSIONS


def load_models(load_detector: bool = True):
    """Load feature extractor, regressor, and optional YOLO detector."""
    global feature_extractor, regressor, feature_type, regressor_model_path, detector_model_path, yolo_detector, yolo_class_names

    feature_model = os.getenv("FEATURE_MODEL", "vit_base")
    logger.info("Loading feature extractor: %s", feature_model)
    feature_extractor = GeneralCrustaceanFeatureExtractor(feature_model)
    feature_type = feature_model

    model_path_env = os.getenv("MODEL_PATH")
    vit_temporal_path = MODELS_DIR / "molt_regressor_vit_temporal.joblib"
    vit_random_forest_path = MODELS_DIR / "molt_regressor_vit_random_forest.joblib"
    best_vit_regressor_path = MODELS_DIR / "best_vit_regressor.joblib"
    temporal_model_path = MODELS_DIR / "temporal" / "Random_Forest_Temporal.pkl"

    if model_path_env:
        model_path = Path(model_path_env)
    elif INFERENCE_MODE == "cpu":
        if best_vit_regressor_path.exists():
            model_path = best_vit_regressor_path
        elif vit_random_forest_path.exists():
            model_path = vit_random_forest_path
        elif vit_temporal_path.exists():
            model_path = vit_temporal_path
        elif temporal_model_path.exists():
            model_path = temporal_model_path
        else:
            raise FileNotFoundError("No compatible VIT model found in models/")
    elif vit_temporal_path.exists():
        model_path = vit_temporal_path
    elif vit_random_forest_path.exists():
        model_path = vit_random_forest_path
    elif best_vit_regressor_path.exists():
        model_path = best_vit_regressor_path
    elif temporal_model_path.exists():
        model_path = temporal_model_path
    else:
        raise FileNotFoundError("No VIT or temporal model found in models/")

    logger.info("Loading regressor from %s", model_path)
    regressor = MoltPhaseRegressor("random_forest")
    regressor.load_model(model_path)
    regressor_model_path = model_path

    if not hasattr(regressor.scaler, "mean_"):
        vit_scaler_path = MODELS_DIR / "vit_scaler.joblib"
        if vit_scaler_path.exists():
            import joblib

            regressor.scaler = joblib.load(vit_scaler_path)
            logger.info("Loaded VIT scaler for %s", model_path.name)

    if load_detector and DETECTION_ENABLED and YOLO and YOLO_MODEL_PATH and YOLO_MODEL_PATH.exists():
        try:
            yolo_detector = YOLO(str(YOLO_MODEL_PATH))
            detector_model_path = YOLO_MODEL_PATH
            if hasattr(yolo_detector, "model") and hasattr(yolo_detector.model, "names"):
                yolo_class_names = yolo_detector.model.names
            logger.info("Loaded YOLO detector for bboxes from %s", YOLO_MODEL_PATH)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Failed to load YOLO detector: %s", exc)

    models_ready.set()


def load_models_async():
    """Load models in a background thread to avoid blocking startup."""
    with model_load_lock:
        if models_ready.is_set():
            return
        try:
            load_models(load_detector=False)
        except Exception as exc:  # pragma: no cover - startup best-effort
            logger.error("Async model load failed: %s", exc)
            return

    if DETECTION_ENABLED and YOLO and YOLO_MODEL_PATH and YOLO_MODEL_PATH.exists():
        try:
            yolo_detector_local = YOLO(str(YOLO_MODEL_PATH))
            globals()["yolo_detector"] = yolo_detector_local
            globals()["detector_model_path"] = YOLO_MODEL_PATH
            if hasattr(yolo_detector_local, "model") and hasattr(yolo_detector_local.model, "names"):
                globals()["yolo_class_names"] = yolo_detector_local.model.names
            logger.info("Loaded YOLO detector in background: %s", YOLO_MODEL_PATH)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Failed to load YOLO detector in background: %s", exc)


def get_molt_phase_category(days_until_molt: float) -> Dict[str, object]:
    """
    Convert days until molt to category and recommendation.

    Updated with refined categories based on field experience:
    - Peeler: < 1 day (harvest immediately)
    - Imminent: < 3 days (monitor closely)
    - Pre-molt Near: < 5 days
    - Pre-molt Later: < 14 days
    - Intermolt: > 14 days
    """
    if days_until_molt < 0:
        return {
            "phase": "Post-molt",
            "color": "#00cc66",
            "recommendation": "Crab has recently molted. Shell is likely soft.",
            "harvest_ready": False,
        }
    if days_until_molt < 1:
        return {
            "phase": "Peeler - Harvest Now",
            "color": "#ff0000",
            "recommendation": "HARVEST IMMEDIATELY! Crab will molt within 24 hours.",
            "harvest_ready": True,
        }
    if days_until_molt < 3:
        return {
            "phase": "Imminent Molt",
            "color": "#ff6600",
            "recommendation": "HARVEST NOW! Crab will molt within 3 days.",
            "harvest_ready": True,
        }
    if days_until_molt < 5:
        return {
            "phase": "Pre-molt (Near)",
            "color": "#c2185b",
            "recommendation": "Monitor closely. Harvest window approaching.",
            "harvest_ready": False,
        }
    if days_until_molt < 14:
        return {
            "phase": "Pre-molt (Early)",
            "color": "#a0ff00",
            "recommendation": "Check again in a few days.",
            "harvest_ready": False,
        }
    return {
        "phase": "Inter-molt",
        "color": "#44ff44",
        "recommendation": "Crab is not close to molting.",
        "harvest_ready": False,
    }


def get_estimated_molt_event_date(days_until_molt: float) -> str:
    """Estimate molt event date relative to the current app date."""
    return (date.today() + timedelta(days=float(days_until_molt))).isoformat()


def get_model_display_name() -> str:
    """Return a human-readable description of the active detector and estimator."""
    detector_name = "fathomnet pretrained detector"
    if detector_model_path and "bootstrapv1" in str(detector_model_path).lower():
        detector_name = "bootstrapv1 detector"

    estimator_name = "transformer based molt estimator"
    if regressor_model_path:
        regressor_name = regressor_model_path.name.lower()
        if "mvp" in regressor_name:
            estimator_name = "mvp v1 estimator"
        elif "temporal" in regressor_name:
            estimator_name = "temporal molt estimator"
        elif "vit" in regressor_name:
            estimator_name = "transformer based molt estimator"

    return f"{detector_name} and {estimator_name}"


def run_detection(image: Image.Image, imgsz: Optional[int] = None) -> List[Dict[str, float]]:
    """Return bbox detections if YOLO detector is configured."""
    if yolo_detector is None:
        return []
    try:
        kwargs = {
            "verbose": False,
            "conf": YOLO_CONF_MIN,
            "max_det": YOLO_MAX_DETECTIONS,
            "iou": YOLO_NMS_IOU,
        }
        if imgsz:
            kwargs["imgsz"] = imgsz
        results = yolo_detector(image, **kwargs)
        boxes_out: List[Dict[str, float]] = []
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else None
                cls = int(box.cls[0]) if box.cls is not None else None
                class_name = None
                if yolo_class_names and cls in yolo_class_names:
                    class_name = yolo_class_names[cls]
                boxes_out.append(
                    {
                        "xmin": xyxy[0],
                        "ymin": xyxy[1],
                        "xmax": xyxy[2],
                        "ymax": xyxy[3],
                        "confidence": conf,
                        "class": cls,
                        "class_name": class_name,
                    }
                )
        return boxes_out
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Detection failed: %s", exc)
        return []


def filter_bboxes(image: Image.Image, bboxes: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Filter bboxes by confidence, area percent, and aspect ratio."""
    if not bboxes:
        return []
    width, height = image.size
    image_area = max(width * height, 1)
    filtered: List[Dict[str, float]] = []
    for box in bboxes:
        class_name = box.get("class_name")
        if class_name and "crab" not in str(class_name).lower():
            continue
        xmin = max(0.0, float(box["xmin"]))
        ymin = max(0.0, float(box["ymin"]))
        xmax = min(float(width), float(box["xmax"]))
        ymax = min(float(height), float(box["ymax"]))
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        if w == 0 or h == 0:
            continue
        conf = float(box.get("confidence") or 0.0)
        area_pct = (w * h) / image_area
        aspect = w / h
        if conf < YOLO_CONF_MIN:
            continue
        if area_pct < YOLO_MIN_AREA_PCT or area_pct > YOLO_MAX_AREA_PCT:
            continue
        if aspect < YOLO_MIN_ASPECT or aspect > YOLO_MAX_ASPECT:
            continue
        filtered.append(box)
    filtered.sort(key=lambda item: item.get("confidence") or 0.0, reverse=True)
    return filtered[:YOLO_MAX_DETECTIONS]


def select_primary_bbox(bboxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Pick the highest-confidence bbox."""
    if not bboxes:
        return None
    return max(bboxes, key=lambda box: box.get("confidence") or 0)


def crop_to_bbox(image: Image.Image, bbox: Dict[str, float]) -> Image.Image:
    """Crop the image to the bbox, clamped to image bounds."""
    width, height = image.size
    xmin = max(0, int(bbox["xmin"]))
    ymin = max(0, int(bbox["ymin"]))
    xmax = min(width, int(bbox["xmax"]))
    ymax = min(height, int(bbox["ymax"]))
    if xmax <= xmin or ymax <= ymin:
        return image
    return image.crop((xmin, ymin, xmax, ymax))


def bbox_iou(first: Optional[Dict[str, float]], second: Optional[Dict[str, float]]) -> float:
    """Compute intersection-over-union for two xyxy bboxes."""
    if not first or not second:
        return 0.0
    x1 = max(float(first["xmin"]), float(second["xmin"]))
    y1 = max(float(first["ymin"]), float(second["ymin"]))
    x2 = min(float(first["xmax"]), float(second["xmax"]))
    y2 = min(float(first["ymax"]), float(second["ymax"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first["xmax"]) - float(first["xmin"])) * max(
        0.0, float(first["ymax"]) - float(first["ymin"])
    )
    second_area = max(0.0, float(second["xmax"]) - float(second["xmin"])) * max(
        0.0, float(second["ymax"]) - float(second["ymin"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def get_fresh_stream_cache(now_ms: int, image: Image.Image) -> Optional[Dict[str, object]]:
    """Return cached stream state while dimensions and TTL still match."""
    with stream_cache_lock:
        cached = dict(stream_cache)
    if not cached:
        return None
    if now_ms - int(cached.get("updated_at_ms", 0)) > STREAM_CACHE_TTL_MS:
        return None
    if cached.get("image_size") != (image.width, image.height):
        return None
    return cached


def update_stream_cache(
    *,
    now_ms: int,
    image: Image.Image,
    bboxes: List[Dict[str, float]],
    primary_bbox: Optional[Dict[str, float]],
    days_until_molt: float,
    phase_info: Dict[str, object],
    estimated_molt_event_date: str,
    recommendation: str,
    estimate_input: str,
) -> None:
    """Store the latest stream result for bbox/estimate reuse."""
    with stream_cache_lock:
        stream_cache.clear()
        stream_cache.update(
            {
                "updated_at_ms": now_ms,
                "estimate_at_ms": now_ms,
                "image_size": (image.width, image.height),
                "bboxes": [dict(box) for box in bboxes],
                "primary_bbox": dict(primary_bbox) if primary_bbox else None,
                "days_until_molt": days_until_molt,
                "estimated_molt_event_date": estimated_molt_event_date,
                "phase_info": dict(phase_info),
                "recommendation": recommendation,
                "estimate_input": estimate_input,
            }
        )


def refresh_stream_cache_detection(
    *,
    now_ms: int,
    image: Image.Image,
    bboxes: List[Dict[str, float]],
    primary_bbox: Optional[Dict[str, float]],
) -> None:
    """Keep stream bbox state current when the heavier estimate is reused."""
    with stream_cache_lock:
        if not stream_cache:
            return
        stream_cache.update(
            {
                "updated_at_ms": now_ms,
                "image_size": (image.width, image.height),
                "bboxes": [dict(box) for box in bboxes],
                "primary_bbox": dict(primary_bbox) if primary_bbox else None,
            }
        )


def encode_thumbnail(
    image: Image.Image,
    label: str,
    bboxes: List[Dict[str, float]],
    bbox_color: str,
) -> str:
    """Create a small JPEG thumbnail with overlay text and optional bbox outlines."""
    thumb = image.copy().convert("RGB")
    orig_w, orig_h = thumb.size
    thumb.thumbnail((320, 320))
    draw = ImageDraw.Draw(thumb)
    color_map = {
        "danger": (220, 53, 69),
        "warning": (255, 193, 7),
        "info": (13, 202, 240),
        "success": (25, 135, 84),
        "primary": (13, 110, 253),
    }
    bbox_rgb = color_map.get(bbox_color, (13, 110, 253))

    # Scale bboxes to thumbnail dimensions
    scale_x = thumb.size[0] / orig_w
    scale_y = thumb.size[1] / orig_h
    for box in bboxes:
        try:
            draw.rectangle(
                [
                    box["xmin"] * scale_x,
                    box["ymin"] * scale_y,
                    box["xmax"] * scale_x,
                    box["ymax"] * scale_y,
                ],
                outline=bbox_rgb,
                width=2,
            )
        except Exception:
            continue

    # Overlay label
    text = label
    text_bg = (255, 255, 255, 200)
    try:
        font = ImageFont.load_default()
        text_size = draw.textbbox((0, 0), text, font=font)
    except Exception:
        font = None
        text_size = (0, 0, len(text) * 6, 12)
    padding = 4
    box_coords = [
        0,
        0,
        text_size[2] + padding * 2,
        text_size[3] + padding * 2,
    ]
    draw.rectangle(box_coords, fill=text_bg)
    draw.text((padding, padding), text, fill="black", font=font)

    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _safe_debug_run_id(run_id: Optional[str]) -> str:
    if run_id:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
        if cleaned:
            return cleaned
    return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"


def parse_aux_tags(
    *,
    aux_tags_json: Optional[str] = None,
    view_angle: Optional[str] = None,
    sex: Optional[str] = None,
    incorrect_detection: Optional[str] = None,
    quality_tag: Optional[str] = None,
    expert_molt_estimate: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge explicit aux fields and optional JSON tags into one review payload."""
    tags: Dict[str, Any] = {}
    if aux_tags_json:
        try:
            parsed = json.loads(aux_tags_json)
            if isinstance(parsed, dict):
                tags.update(parsed)
            else:
                tags["aux_tags_json"] = parsed
        except json.JSONDecodeError:
            tags["aux_tags_json_error"] = aux_tags_json

    for key, value in {
        "view_angle": view_angle,
        "sex": sex,
        "incorrect_detection": incorrect_detection,
        "quality_tag": quality_tag,
        "expert_molt_estimate": expert_molt_estimate,
        "review_notes": review_notes,
    }.items():
        if value not in (None, ""):
            tags[key] = value
    return tags


def render_debug_overlay(
    image: Image.Image,
    *,
    bboxes: List[Dict[str, float]],
    primary_bbox: Optional[Dict[str, float]],
    phase_info: Dict[str, object],
    days_until_molt: float,
    estimate_input: str,
    aux_tags: Dict[str, Any],
) -> Image.Image:
    """Render a review overlay for the debug export bundle."""
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    width, height = overlay.size
    panel_lines = [
        f"{get_model_display_name()}",
        f"{phase_info.get('phase', 'N/A')} | {days_until_molt:.1f}d",
        f"input: {estimate_input}",
    ]
    if aux_tags:
        tag_bits = [f"{key}={value}" for key, value in aux_tags.items() if value not in (None, "")]
        if tag_bits:
            panel_lines.append("tags: " + ", ".join(tag_bits[:3]))

    panel_text = "\n".join(panel_lines)
    try:
        bbox = draw.multiline_textbbox((0, 0), panel_text, font=font, spacing=3)
        panel_w = min(width - 16, bbox[2] - bbox[0] + 20)
        panel_h = bbox[3] - bbox[1] + 20
    except Exception:
        panel_w = min(width - 16, 520)
        panel_h = 88
    draw.rounded_rectangle([8, 8, 8 + panel_w, 8 + panel_h], radius=10, fill=(255, 255, 255), outline=(13, 110, 253), width=2)
    draw.multiline_text((18, 16), panel_text, fill=(20, 24, 28), font=font, spacing=3)

    for idx, box in enumerate(bboxes, start=1):
        try:
            xmin = float(box["xmin"])
            ymin = float(box["ymin"])
            xmax = float(box["xmax"])
            ymax = float(box["ymax"])
            is_primary = primary_bbox is not None and bbox_iou(box, primary_bbox) >= 0.99
            color = (13, 110, 253) if is_primary else (32, 201, 151)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=6 if is_primary else 4)
            label = f"#{idx} {float(box.get('confidence') or 0.0):.2f}"
            if box.get("class_name"):
                label += f" {box['class_name']}"
            label_y = ymin - 24 if ymin > 28 else ymin + 4
            draw.rectangle([xmin, label_y, xmin + 12 + 8 * len(label), label_y + 22], fill=color)
            draw.text((xmin + 6, label_y + 4), label, fill="white", font=font)
        except Exception:
            continue
    return overlay


def save_debug_export(
    *,
    raw_bytes: bytes,
    source_filename: str,
    image: Image.Image,
    roi_image: Image.Image,
    bboxes: List[Dict[str, float]],
    primary_bbox: Optional[Dict[str, float]],
    raw_bboxes: List[Dict[str, float]],
    phase_info: Dict[str, object],
    days_until_molt: float,
    estimated_molt_event_date: str,
    recommendation: str,
    estimate_input: str,
    aux_tags: Dict[str, Any],
    response_payload: Dict[str, object],
    debug_run_id: Optional[str],
) -> Dict[str, Any]:
    """Persist a review bundle for later expert inspection."""
    run_id = _safe_debug_run_id(debug_run_id)
    frame_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    frame_dir = DEBUG_EXPORTS_DIR / run_id / frame_id
    crops_dir = frame_dir / "crops"
    frame_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_suffix = Path(source_filename).suffix.lower()
    if source_suffix not in UPLOAD_EXTENSIONS:
        source_suffix = ".bin"
    source_input_path = frame_dir / f"source_input{source_suffix}"
    source_input_path.write_bytes(raw_bytes)

    input_path = frame_dir / "input.jpg"
    image.convert("RGB").save(input_path, quality=92)

    detections_overlay = render_debug_overlay(
        image,
        bboxes=bboxes,
        primary_bbox=primary_bbox,
        phase_info=phase_info,
        days_until_molt=days_until_molt,
        estimate_input=estimate_input,
        aux_tags=aux_tags,
    )
    detections_path = frame_dir / "detections.jpg"
    detections_overlay.save(detections_path, quality=92)

    estimate_input_path = frame_dir / "estimate_input.jpg"
    roi_image.convert("RGB").save(estimate_input_path, quality=92)

    crop_paths: List[str] = []
    for idx, bbox in enumerate(bboxes, start=1):
        crop_path = crops_dir / f"bbox_{idx:02d}.jpg"
        try:
            xmin = max(0, int(float(bbox["xmin"])))
            ymin = max(0, int(float(bbox["ymin"])))
            xmax = min(image.width, int(float(bbox["xmax"])))
            ymax = min(image.height, int(float(bbox["ymax"])))
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop.save(crop_path, quality=92)
            crop_paths.append(str(crop_path.relative_to(DEBUG_EXPORTS_DIR / run_id)))
        except Exception:
            continue

    metadata = {
        "run_id": run_id,
        "frame_id": frame_id,
        "source_filename": source_filename,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_width": image.width,
        "image_height": image.height,
        "days_until_molt": days_until_molt,
        "estimated_molt_event_date": estimated_molt_event_date,
        "recommendation": recommendation,
        "estimate_input": estimate_input,
        "phase_info": phase_info,
        "aux_tags": aux_tags,
        "primary_bbox": primary_bbox,
        "bboxes": bboxes,
        "raw_bboxes": raw_bboxes,
        "response": response_payload,
        "files": {
            "source_input": source_input_path.name,
            "input": "input.jpg",
            "detections": "detections.jpg",
            "estimate_input": "estimate_input.jpg",
            "crops": crop_paths,
        },
    }

    metadata_path = frame_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest_path = DEBUG_EXPORTS_DIR / run_id / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "frame_id": frame_id,
            "source_filename": source_filename,
            "frame_dir": str(frame_dir.relative_to(DEBUG_EXPORTS_DIR / run_id)),
            "metadata": "metadata.json",
        }) + "\n")

    return {
        "enabled": True,
        "run_id": run_id,
        "frame_id": frame_id,
        "frame_dir": str(frame_dir.relative_to(DEBUG_EXPORTS_DIR)),
        "metadata_path": str(metadata_path.relative_to(DEBUG_EXPORTS_DIR)),
        "input_path": str(input_path.relative_to(DEBUG_EXPORTS_DIR)),
        "detections_path": str(detections_path.relative_to(DEBUG_EXPORTS_DIR)),
        "estimate_input_path": str(estimate_input_path.relative_to(DEBUG_EXPORTS_DIR)),
    }


def build_debug_export_zip(run_id: str) -> Path:
    """Zip a debug export run for download."""
    safe_run_id = _safe_debug_run_id(run_id)
    run_dir = DEBUG_EXPORTS_DIR / safe_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Debug export run not found: {safe_run_id}")
    zip_path = DEBUG_EXPORTS_DIR / f"{safe_run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(run_dir)))
    return zip_path


def _safe_debug_session_name(run_name: Optional[str]) -> str:
    if not run_name:
        return _safe_debug_run_id(None)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._-")
    return cleaned or _safe_debug_run_id(None)


def _write_debug_session_state() -> None:
    DEBUG_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_SESSION_DIR / "session.json").write_text(json.dumps(debug_session_state, indent=2), encoding="utf-8")


def _reset_debug_session_dir() -> None:
    if DEBUG_SESSION_DIR.exists():
        shutil.rmtree(DEBUG_SESSION_DIR)
    DEBUG_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    for zip_path in DEBUG_EXPORTS_DIR.glob("moltmeter_debug_*.zip"):
        try:
            zip_path.unlink()
        except OSError:
            continue


def _load_debug_session_manifest() -> List[Dict[str, Any]]:
    manifest_path = DEBUG_SESSION_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _normalize_molt_details(molt_details: Optional[Any]) -> List[str]:
    if molt_details is None:
        return []
    if isinstance(molt_details, list):
        return [str(value) for value in molt_details if str(value).strip()]
    if isinstance(molt_details, str):
        try:
            parsed = json.loads(molt_details)
        except json.JSONDecodeError:
            return [part.strip() for part in molt_details.split(",") if part.strip()]
        if isinstance(parsed, list):
            return [str(value) for value in parsed if str(value).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    return []


def build_debug_session_name(location_name: str, started_at_utc: Optional[str] = None) -> str:
    timestamp = started_at_utc or time.strftime("%Y-%m-%d_%H%M%SZ", time.gmtime())
    location_bits = re.sub(r"[^A-Za-z0-9]+", "_", (location_name or "unknown")).strip("_")
    return f"{timestamp}_{location_bits or 'unknown'}"


def start_debug_session(run_name: str, location_name: str) -> Dict[str, Any]:
    session_id = _safe_debug_session_name(run_name)
    with debug_session_lock:
        _reset_debug_session_dir()
        debug_session_state.update(
            {
                "active": True,
                "session_id": session_id,
                "run_name": run_name,
                "location_name": location_name,
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ended_at_utc": None,
                "capture_count": 0,
            }
        )
        _write_debug_session_state()
    return dict(debug_session_state)


def update_debug_session(location_name: str) -> Dict[str, Any]:
    with debug_session_lock:
        if not debug_session_state.get("session_id"):
            raise RuntimeError("No active debug session")
        debug_session_state["location_name"] = location_name
        debug_session_state["run_name"] = build_debug_session_name(
            location_name,
            debug_session_state.get("started_at_utc"),
        )
        _write_debug_session_state()
        return dict(debug_session_state)


def stop_debug_session() -> Dict[str, Any]:
    with debug_session_lock:
        if debug_session_state.get("session_id"):
            debug_session_state["active"] = False
            debug_session_state["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_debug_session_state()
    return dict(debug_session_state)


def get_active_debug_session() -> Optional[Dict[str, Any]]:
    with debug_session_lock:
        if not debug_session_state.get("session_id"):
            return None
        return dict(debug_session_state)


def _make_bbox_thumbnail(image: Image.Image, bbox: Dict[str, Any], thumb_path: Path) -> None:
    xmin = max(0, int(float(bbox.get("xmin", 0))))
    ymin = max(0, int(float(bbox.get("ymin", 0))))
    xmax = min(image.width, int(float(bbox.get("xmax", image.width))))
    ymax = min(image.height, int(float(bbox.get("ymax", image.height))))
    crop = image.crop((xmin, ymin, xmax, ymax)).convert("RGB")
    crop.thumbnail((220, 220))
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(thumb_path, quality=90)


def save_debug_session_capture(
    *,
    raw_bytes: bytes,
    source_filename: str,
    image: Image.Image,
    roi_image: Image.Image,
    bboxes: List[Dict[str, float]],
    primary_bbox: Optional[Dict[str, float]],
    raw_bboxes: List[Dict[str, float]],
    phase_info: Dict[str, object],
    days_until_molt: float,
    estimated_molt_event_date: str,
    recommendation: str,
    estimate_input: str,
    aux_tags: Dict[str, Any],
    response_payload: Dict[str, object],
    molt_details: List[str],
) -> Dict[str, Any]:
    """Persist a capture into the active expert-review session."""
    session = get_active_debug_session()
    if not session or not session.get("active"):
        raise RuntimeError("No active debug session")

    capture_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    capture_dir = DEBUG_SESSION_DIR / "captures" / capture_id
    crops_dir = capture_dir / "crops"
    capture_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_suffix = Path(source_filename).suffix.lower()
    if source_suffix not in UPLOAD_EXTENSIONS:
        source_suffix = ".bin"
    source_input_path = capture_dir / f"source_input{source_suffix}"
    source_input_path.write_bytes(raw_bytes)
    source_input_relpath = str(source_input_path.relative_to(DEBUG_SESSION_DIR))

    input_path = capture_dir / "input.jpg"
    image.convert("RGB").save(input_path, quality=92)

    detections_overlay = render_debug_overlay(
        image,
        bboxes=bboxes,
        primary_bbox=primary_bbox,
        phase_info=phase_info,
        days_until_molt=days_until_molt,
        estimate_input=estimate_input,
        aux_tags=aux_tags,
    )
    detections_path = capture_dir / "detections.jpg"
    detections_overlay.save(detections_path, quality=92)

    estimate_input_path = capture_dir / "estimate_input.jpg"
    roi_image.convert("RGB").save(estimate_input_path, quality=92)

    bbox_thumbnail_path = capture_dir / "bbox_thumbnail.jpg"
    if primary_bbox:
        _make_bbox_thumbnail(image, primary_bbox, bbox_thumbnail_path)
    else:
        fallback_thumb = image.copy().convert("RGB")
        fallback_thumb.thumbnail((220, 220))
        fallback_thumb.save(bbox_thumbnail_path, quality=90)

    crop_paths: List[str] = []
    for idx, bbox in enumerate(bboxes, start=1):
        crop_path = crops_dir / f"bbox_{idx:02d}.jpg"
        try:
            xmin = max(0, int(float(bbox["xmin"])))
            ymin = max(0, int(float(bbox["ymin"])))
            xmax = min(image.width, int(float(bbox["xmax"])))
            ymax = min(image.height, int(float(bbox["ymax"])))
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop.save(crop_path, quality=92)
            crop_paths.append(str(crop_path.relative_to(DEBUG_SESSION_DIR)))
        except Exception:
            continue

    capture_metadata = {
        "session_id": session["session_id"],
        "session_name": session["run_name"],
        "location_name": session["location_name"],
        "capture_id": capture_id,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_filename": source_filename,
        "source_file_path": source_input_relpath,
        "image_width": image.width,
        "image_height": image.height,
        "days_until_molt": days_until_molt,
        "estimated_molt_event_date": estimated_molt_event_date,
        "recommendation": recommendation,
        "estimate_input": estimate_input,
        "phase_info": phase_info,
        "aux_tags": aux_tags,
        "molt_details": molt_details,
        "primary_bbox": primary_bbox,
        "bboxes": bboxes,
        "raw_bboxes": raw_bboxes,
        "response": response_payload,
        "files": {
            "source_input": source_input_path.name,
            "input": "input.jpg",
            "detections": "detections.jpg",
            "estimate_input": "estimate_input.jpg",
            "bbox_thumbnail": "bbox_thumbnail.jpg",
            "crops": crop_paths,
        },
    }

    metadata_path = capture_dir / "metadata.json"
    metadata_path.write_text(json.dumps(capture_metadata, indent=2), encoding="utf-8")

    manifest_path = DEBUG_SESSION_DIR / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "capture_id": capture_id,
                    "captured_at_utc": capture_metadata["captured_at_utc"],
                    "source_filename": source_filename,
                    "capture_dir": str(capture_dir.relative_to(DEBUG_SESSION_DIR)),
                    "metadata": "metadata.json",
                }
            )
            + "\n"
        )

    with debug_session_lock:
        debug_session_state["capture_count"] = int(debug_session_state.get("capture_count") or 0) + 1
        _write_debug_session_state()

    return {
        "capture_id": capture_id,
        "capture_dir": str(capture_dir.relative_to(DEBUG_SESSION_DIR)),
        "metadata_path": str(metadata_path.relative_to(DEBUG_SESSION_DIR)),
        "input_path": str(input_path.relative_to(DEBUG_SESSION_DIR)),
        "detections_path": str(detections_path.relative_to(DEBUG_SESSION_DIR)),
        "estimate_input_path": str(estimate_input_path.relative_to(DEBUG_SESSION_DIR)),
        "bbox_thumbnail_path": str(bbox_thumbnail_path.relative_to(DEBUG_SESSION_DIR)),
    }


def build_debug_session_workbook(session_dir: Path) -> Path:
    """Create an Excel workbook for all captures in the active debug session."""
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Font
    except Exception as exc:  # pragma: no cover - dependency guarded at runtime
        raise RuntimeError("openpyxl is required for debug spreadsheet exports") from exc

    workbook_path = session_dir / "captures.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Captures"
    headers = [
        "Capture ID",
        "Captured At UTC",
        "Session Name",
        "Location",
        "Source File",
        "View",
        "Sex",
        "Incorrect Detection",
        "Molt Details",
        "Expert Molt Estimate",
        "Notes",
        "BBox Count",
        "Days Until Molt",
        "Phase",
        "Recommendation",
        "Primary Confidence",
        "Thumbnail",
    ]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center")

    width_map = {
        "A": 26,
        "B": 22,
        "C": 34,
        "D": 18,
        "E": 28,
        "F": 12,
        "G": 12,
        "H": 16,
        "I": 26,
        "J": 22,
        "K": 28,
        "L": 12,
        "M": 15,
        "N": 18,
        "O": 36,
        "P": 16,
        "Q": 20,
    }
    for col, width in width_map.items():
        sheet.column_dimensions[col].width = width

    manifest_rows = _load_debug_session_manifest()
    session_state_path = session_dir / "session.json"
    session_state: Dict[str, Any] = {}
    if session_state_path.exists():
        try:
            session_state = json.loads(session_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session_state = {}
    session_name = session_state.get("run_name")
    session_location = session_state.get("location_name")
    for row_index, manifest_row in enumerate(manifest_rows, start=2):
        capture_dir = session_dir / manifest_row["capture_dir"]
        metadata_path = capture_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        aux_tags = metadata.get("aux_tags", {})
        primary_bbox = metadata.get("primary_bbox") or {}
        thumbnail_path = capture_dir / "bbox_thumbnail.jpg"
        sheet.append(
            [
                metadata.get("capture_id", ""),
                metadata.get("captured_at_utc", ""),
                session_name or metadata.get("session_name", ""),
                session_location or metadata.get("location_name", ""),
                metadata.get("source_file_path", metadata.get("source_filename", "")),
                aux_tags.get("view_angle", "unknown"),
                aux_tags.get("sex", "unknown"),
                aux_tags.get("incorrect_detection", "none"),
                ", ".join(metadata.get("molt_details", [])),
                aux_tags.get("expert_molt_estimate", ""),
                aux_tags.get("review_notes", ""),
                len(metadata.get("bboxes", [])),
                metadata.get("days_until_molt", ""),
                metadata.get("phase_info", {}).get("phase", ""),
                metadata.get("recommendation", ""),
                primary_bbox.get("confidence", ""),
                "",
            ]
        )
        sheet.row_dimensions[row_index].height = 96
        if thumbnail_path.exists():
            try:
                img = XLImage(str(thumbnail_path))
                img.width = 88
                img.height = 88
                sheet.add_image(img, f"Q{row_index}")
            except Exception:
                continue

    workbook.save(workbook_path)
    return workbook_path


def _debug_session_rows(session_dir: Path) -> List[List[Any]]:
    """Build spreadsheet-compatible rows for debug/session captures."""
    manifest_rows = _load_debug_session_manifest()
    session_state_path = session_dir / "session.json"
    session_state: Dict[str, Any] = {}
    if session_state_path.exists():
        try:
            session_state = json.loads(session_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session_state = {}
    session_name = session_state.get("run_name")
    session_location = session_state.get("location_name")
    rows: List[List[Any]] = []
    for manifest_row in manifest_rows:
        capture_dir = session_dir / manifest_row["capture_dir"]
        metadata_path = capture_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        aux_tags = metadata.get("aux_tags", {})
        primary_bbox = metadata.get("primary_bbox") or {}
        rows.append(
            [
                metadata.get("capture_id", ""),
                metadata.get("captured_at_utc", ""),
                session_name or metadata.get("session_name", ""),
                session_location or metadata.get("location_name", ""),
                metadata.get("source_file_path", metadata.get("source_filename", "")),
                aux_tags.get("view_angle", "unknown"),
                aux_tags.get("sex", "unknown"),
                aux_tags.get("incorrect_detection", "none"),
                ", ".join(metadata.get("molt_details", [])),
                aux_tags.get("expert_molt_estimate", ""),
                aux_tags.get("review_notes", ""),
                len(metadata.get("bboxes", [])),
                metadata.get("days_until_molt", ""),
                metadata.get("phase_info", {}).get("phase", ""),
                metadata.get("recommendation", ""),
                primary_bbox.get("confidence", ""),
                metadata.get("files", {}).get("bbox_thumbnail", ""),
            ]
        )
    return rows


def build_debug_session_csv(session_dir: Path) -> Path:
    """Create a dependency-free CSV summary for all captures."""
    import csv

    csv_path = session_dir / "captures.csv"
    headers = [
        "Capture ID",
        "Captured At UTC",
        "Session Name",
        "Location",
        "Source File",
        "View",
        "Sex",
        "Incorrect Detection",
        "Molt Details",
        "Expert Molt Estimate",
        "Notes",
        "BBox Count",
        "Days Until Molt",
        "Phase",
        "Recommendation",
        "Primary Confidence",
        "Thumbnail",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(_debug_session_rows(session_dir))
    return csv_path


def build_debug_session_zip() -> Path:
    """Zip the active debug session for download."""
    session = get_active_debug_session()
    if not session or not session.get("session_id"):
        raise FileNotFoundError("No active debug session")
    if not DEBUG_SESSION_DIR.exists():
        raise FileNotFoundError("No active debug session data found")
    try:
        build_debug_session_workbook(DEBUG_SESSION_DIR)
    except RuntimeError as exc:
        logger.warning("Excel debug export unavailable, writing CSV fallback: %s", exc)
        build_debug_session_csv(DEBUG_SESSION_DIR)
    zip_name = f"moltmeter_debug_{_safe_debug_run_id(session.get('run_name'))}.zip"
    zip_path = DEBUG_EXPORTS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in DEBUG_SESSION_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(DEBUG_SESSION_DIR)))
    return zip_path


async def predict_image(
    file: UploadFile,
    *,
    stream_mode: bool = False,
    include_thumbnail: bool = True,
    detection_imgsz: Optional[int] = None,
    debug_export: bool = False,
    debug_run_id: Optional[str] = None,
    aux_tags: Optional[Dict[str, Any]] = None,
    capture_debug: bool = False,
    molt_details: Optional[List[str]] = None,
) -> Dict[str, object]:
    started_at = time.perf_counter()
    stage_started_at = started_at
    timing_ms: Dict[str, float] = {}

    def mark(stage: str) -> None:
        nonlocal stage_started_at
        now = time.perf_counter()
        timing_ms[stage] = round((now - stage_started_at) * 1000, 1)
        stage_started_at = now

    if not models_ready.is_set():
        deadline = asyncio.get_running_loop().time() + float(os.getenv("MODEL_READY_TIMEOUT", "20"))
        while not models_ready.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.2)
        if not models_ready.is_set():
            raise HTTPException(status_code=503, detail="Model loading, please retry shortly.")
    mark("model_wait")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    raw_bytes = await file.read()
    mark("upload_read")
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc
    mark("decode")

    raw_bboxes = run_detection(image, imgsz=detection_imgsz) if DETECTION_ENABLED else []
    bboxes = filter_bboxes(image, raw_bboxes) if DETECTION_ENABLED else []
    now_ms = int(time.time() * 1000)
    cached = get_fresh_stream_cache(now_ms, image) if stream_mode else None
    bbox_cached = False
    if stream_mode and not bboxes and cached and cached.get("bboxes"):
        cached_age_ms = now_ms - int(cached.get("updated_at_ms", 0))
        if cached_age_ms <= STREAM_CACHE_TTL_MS:
            bboxes = [dict(box) for box in cached.get("bboxes", [])]
            bbox_cached = True
    primary_bbox = select_primary_bbox(bboxes) if DETECTION_CROP_ENABLED else None
    roi_image = crop_to_bbox(image, primary_bbox) if primary_bbox else image
    estimate_input = "yolo_crop" if primary_bbox else "whole_image_fallback" if DETECTION_ENABLED else "whole_image"
    mark("detect")

    estimate_cached = False
    should_reuse_estimate = False
    if stream_mode and cached:
        cached_primary = cached.get("primary_bbox") if isinstance(cached.get("primary_bbox"), dict) else None
        same_bbox = primary_bbox is not None and bbox_iou(primary_bbox, cached_primary) >= STREAM_BBOX_REUSE_IOU
        estimate_age_ms = now_ms - int(cached.get("estimate_at_ms", 0))
        should_reuse_estimate = same_bbox and estimate_age_ms < STREAM_ESTIMATE_EVERY_MS

    if should_reuse_estimate and cached:
        days_until_molt = float(cached["days_until_molt"])
        estimated_molt_event_date = str(cached["estimated_molt_event_date"])
        phase_info = dict(cached["phase_info"])  # type: ignore[arg-type]
        recommendation = str(cached["recommendation"])
        estimate_input = str(cached["estimate_input"])
        estimate_cached = True
        refresh_stream_cache_detection(
            now_ms=now_ms,
            image=image,
            bboxes=bboxes,
            primary_bbox=primary_bbox,
        )
    else:
        if feature_extractor is None or regressor is None:
            raise HTTPException(status_code=503, detail="Models are not loaded.")
        np_image = np.array(roi_image)
        features = feature_extractor.extract_features(np_image).reshape(1, -1)
        days_until_molt = float(regressor.predict(features)[0])
        estimated_molt_event_date = get_estimated_molt_event_date(days_until_molt)
        phase_info = get_molt_phase_category(days_until_molt)
        recommendation = str(phase_info["recommendation"])
        if DETECTION_ENABLED and not bboxes:
            recommendation = (
                "No confident crab detection. Estimate was run on the full image; "
                "try a closer, centered shot for a crop-based estimate."
            )
        if stream_mode:
            update_stream_cache(
                now_ms=now_ms,
                image=image,
                bboxes=bboxes,
                primary_bbox=primary_bbox,
                days_until_molt=days_until_molt,
                phase_info=phase_info,
                estimated_molt_event_date=estimated_molt_event_date,
                recommendation=recommendation,
                estimate_input=estimate_input,
            )
    mark("estimate" if not estimate_cached else "estimate_cache")

    label_text = f"{phase_info['phase']} ({days_until_molt:.1f}d)"
    thumbnail = None
    if include_thumbnail:
        thumbnail = encode_thumbnail(image, label_text, bboxes, str(phase_info["color"]))
    mark("thumbnail" if include_thumbnail else "thumbnail_skip")
    timing_ms["total"] = round((time.perf_counter() - started_at) * 1000, 1)

    debug_export_payload = None
    if debug_export:
        debug_export_payload = save_debug_export(
            raw_bytes=raw_bytes,
            source_filename=file.filename or "upload.jpg",
            image=image,
            roi_image=roi_image,
            bboxes=bboxes,
            primary_bbox=primary_bbox,
            raw_bboxes=raw_bboxes,
            phase_info=phase_info,
            days_until_molt=days_until_molt,
            estimated_molt_event_date=estimated_molt_event_date,
            recommendation=recommendation,
            estimate_input=estimate_input,
            aux_tags=aux_tags or {},
            response_payload={
                "days_until_molt": days_until_molt,
                "estimated_molt_event_date": estimated_molt_event_date,
                "phase": phase_info["phase"],
                "recommendation": recommendation,
                "app_estimate_input": estimate_input,
                "primary_bbox": primary_bbox,
                "bboxes": bboxes,
            },
            debug_run_id=debug_run_id,
        )

    debug_capture_payload = None
    if capture_debug:
        debug_capture_payload = save_debug_session_capture(
            raw_bytes=raw_bytes,
            source_filename=file.filename or "upload.jpg",
            image=image,
            roi_image=roi_image,
            bboxes=bboxes,
            primary_bbox=primary_bbox,
            raw_bboxes=raw_bboxes,
            phase_info=phase_info,
            days_until_molt=days_until_molt,
            estimated_molt_event_date=estimated_molt_event_date,
            recommendation=recommendation,
            estimate_input=estimate_input,
            aux_tags=aux_tags or {},
            response_payload={
                "days_until_molt": days_until_molt,
                "estimated_molt_event_date": estimated_molt_event_date,
                "phase": phase_info["phase"],
                "recommendation": recommendation,
                "app_estimate_input": estimate_input,
                "primary_bbox": primary_bbox,
                "bboxes": bboxes,
            },
            molt_details=molt_details or [],
        )

    return {
        "success": True,
        "days_until_molt": days_until_molt,
        "estimated_molt_event_date": estimated_molt_event_date,
        "phase": phase_info["phase"],
        "color": phase_info["color"],
        "recommendation": recommendation,
        "harvest_ready": phase_info["harvest_ready"],
        "confidence": "High" if abs(days_until_molt) < 20 else "Medium",
        "app_estimate_input": estimate_input,
        "whole_image_fallback_used": DETECTION_ENABLED and primary_bbox is None,
        "thumbnail": thumbnail,
        "feature_type": feature_type.upper() if feature_type else None,
        "model_display_name": get_model_display_name(),
        "bbox_count": len(bboxes),
        "crop_used": primary_bbox is not None,
        "stream_mode": stream_mode,
        "bbox_cached": bbox_cached,
        "estimate_cached": estimate_cached,
        "primary_bbox": primary_bbox,
        "bboxes": bboxes,
        "image_width": image.width,
        "image_height": image.height,
        "aux_tags": aux_tags or {},
        "debug_export": debug_export_payload,
        "debug_capture": debug_capture_payload,
        "server_timing_ms": timing_ms,
        "detection_debug": {
            "enabled": DETECTION_ENABLED,
            "raw_count": len(raw_bboxes),
            "filtered_count": len(bboxes),
            "stream_imgsz": detection_imgsz,
            "class_filter": "Crab",
            "filters": {
                "conf_min": YOLO_CONF_MIN,
                "max_detections": YOLO_MAX_DETECTIONS,
                "nms_iou": YOLO_NMS_IOU,
                "min_area_pct": YOLO_MIN_AREA_PCT,
                "max_area_pct": YOLO_MAX_AREA_PCT,
                "min_aspect": YOLO_MIN_ASPECT,
                "max_aspect": YOLO_MAX_ASPECT,
            },
            "raw_bboxes": raw_bboxes[:3],
            "filtered_bboxes": bboxes[:3],
        },
    }


@app.on_event("startup")
def startup_event():
    async_load = os.getenv("MODEL_LOAD_ASYNC", "true").lower() == "true"
    if async_load:
        threading.Thread(target=load_models_async, daemon=True).start()
        logger.info("Model loading started in background.")
        return
    load_models()
    if feature_extractor is None or regressor is None:
        logger.error("Failed to load models on startup")
        raise RuntimeError("Models not loaded")
    logger.info("Models loaded. FastAPI app ready.")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "feature_extractor": feature_extractor is not None,
        "regressor": regressor is not None and regressor.is_fitted if regressor else False,
        "feature_type": feature_type,
        "model_display_name": get_model_display_name() if regressor is not None else None,
        "yolo_detector": yolo_detector is not None,
        "models_ready": models_ready.is_set(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    aux_tags_json: Optional[str] = Form(None),
    view_angle: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    incorrect_detection: Optional[str] = Form(None),
    quality_tag: Optional[str] = Form(None),
    expert_molt_estimate: Optional[str] = Form(None),
    review_notes: Optional[str] = Form(None),
):
    async with inference_semaphore:
        try:
            result = await predict_image(
                file,
                aux_tags=parse_aux_tags(
                    aux_tags_json=aux_tags_json,
                    view_angle=view_angle,
                    sex=sex,
                    incorrect_detection=incorrect_detection,
                    quality_tag=quality_tag,
                    expert_molt_estimate=expert_molt_estimate,
                    review_notes=review_notes,
                ),
            )
            return JSONResponse(content=result)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict_stream")
async def predict_stream(
    file: UploadFile = File(...),
    aux_tags_json: Optional[str] = Form(None),
    view_angle: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    incorrect_detection: Optional[str] = Form(None),
    quality_tag: Optional[str] = Form(None),
    expert_molt_estimate: Optional[str] = Form(None),
    export_capture: bool = Form(False),
    review_notes: Optional[str] = Form(None),
):
    async with inference_semaphore:
        try:
            aux_tags = parse_aux_tags(
                aux_tags_json=aux_tags_json,
                view_angle=view_angle,
                sex=sex,
                incorrect_detection=incorrect_detection,
                quality_tag=quality_tag,
                expert_molt_estimate=expert_molt_estimate,
                review_notes=review_notes,
            )
            result = await predict_image(
                file,
                stream_mode=True,
                include_thumbnail=not STREAM_SKIP_THUMBNAIL,
                detection_imgsz=STREAM_YOLO_IMGSZ,
                aux_tags=aux_tags,
                capture_debug=export_capture,
                molt_details=_normalize_molt_details(aux_tags.get("molt_details")),
            )
            if export_capture:
                result["debug_session"] = get_active_debug_session() or dict(debug_session_state)
            return JSONResponse(content=result)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.exception("Streaming prediction failed")
            raise HTTPException(status_code=500, detail=f"Streaming prediction failed: {exc}") from exc


@app.get("/debug-exports/{run_id}")
def download_debug_export(run_id: str):
    try:
        zip_path = build_debug_export_zip(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"moltmeter_debug_{_safe_debug_run_id(run_id)}.zip",
    )


@app.post("/debug-session/start")
def debug_session_start(
    run_name: Optional[str] = Form(None),
    location_name: Optional[str] = Form(None),
):
    resolved_location = (location_name or "local").strip() or "local"
    resolved_run_name = (run_name or build_debug_session_name(resolved_location)).strip()
    state = start_debug_session(resolved_run_name, resolved_location)
    return JSONResponse(content={"success": True, "session": state})


@app.post("/debug-session/stop")
def debug_session_stop():
    state = stop_debug_session()
    return JSONResponse(content={"success": True, "session": state})


@app.post("/debug-session/update")
def debug_session_update(
    location_name: Optional[str] = Form(None),
):
    resolved_location = (location_name or "local").strip() or "local"
    try:
        state = update_debug_session(resolved_location)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content={"success": True, "session": state})


@app.get("/debug-session/status")
def debug_session_status():
    return JSONResponse(content={"success": True, "session": get_active_debug_session() or dict(debug_session_state)})


@app.post("/debug-session/capture")
async def debug_session_capture(
    file: UploadFile = File(...),
    aux_tags_json: Optional[str] = Form(None),
    view_angle: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    incorrect_detection: Optional[str] = Form(None),
    molt_details_json: Optional[str] = Form(None),
    quality_tag: Optional[str] = Form(None),
    expert_molt_estimate: Optional[str] = Form(None),
    review_notes: Optional[str] = Form(None),
):
    async with inference_semaphore:
        session = get_active_debug_session()
        if not session or not session.get("active"):
            raise HTTPException(status_code=400, detail="No active debug session")
        try:
            result = await predict_image(
                file,
                aux_tags=parse_aux_tags(
                    aux_tags_json=aux_tags_json,
                    view_angle=view_angle,
                    sex=sex,
                    incorrect_detection=incorrect_detection,
                    quality_tag=quality_tag,
                    expert_molt_estimate=expert_molt_estimate,
                    review_notes=review_notes,
                ),
                capture_debug=True,
                molt_details=_normalize_molt_details(molt_details_json),
            )
            result["debug_session"] = get_active_debug_session() or session
            return JSONResponse(content=result)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.exception("Debug capture failed")
            raise HTTPException(status_code=500, detail=f"Debug capture failed: {exc}") from exc


@app.get("/debug-session/download")
def debug_session_download():
    try:
        zip_path = build_debug_session_zip()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session = get_active_debug_session() or {}
    session_name = session.get("run_name") or "debug_session"
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"moltmeter_debug_{_safe_debug_run_id(session_name)}.zip",
    )


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the web UI at the root domain."""
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse)
def ui():
    """Alias for root - serves the same web UI."""
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/api")
def api_info():
    """API information endpoint."""
    return {
        "message": "Green Crab Molt Detector API (FastAPI)",
        "endpoints": [
            "/predict",
            "/predict_stream",
            "/debug-exports/{run_id}",
            "/debug-session/start",
            "/debug-session/stop",
            "/debug-session/status",
            "/debug-session/capture",
            "/debug-session/download",
            "/health",
        ],
    }


@app.get("/about")
def about():
    """About page for web filters and categorization services."""
    return {
        "name": "MoltMeter - Green Crab Molt Detection",
        "description": "AI-powered tool for predicting green crab molt phases to support sustainable harvesting",
        "purpose": "Marine biology research and sustainable fisheries management",
        "technology": "Computer vision and machine learning for molt phase estimation",
        "category": "Science and Research / Environmental / Sustainable Fisheries",
        "contact": "https://moltmeter.ai",
        "privacy": "No personal data collected. Only crab images processed for molt prediction.",
        "security": "Hosted on Google Cloud Platform with SSL encryption"
    }


@app.get("/about-page", response_class=HTMLResponse)
async def about_page():
    """Serve the About Us page."""
    about_path = BASE_PATH / "templates" / "about.html"
    if not about_path.exists():
        raise HTTPException(status_code=404, detail="About page not found")
    return HTMLResponse(about_path.read_text(encoding="utf-8"))


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    """Serve robots.txt for search engines."""
    with open(BASE_PATH / "static" / "robots.txt") as f:
        return f.read()
