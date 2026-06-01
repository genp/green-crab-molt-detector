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
import logging
import os
import sys
import threading
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

# Patch for torch/transformers compatibility issue (mirrors flask app)
import torch

if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32
if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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
YOLO_CONF_MIN = float(os.getenv("YOLO_CONF_MIN", "0.2"))
YOLO_MIN_AREA_PCT = float(os.getenv("YOLO_MIN_AREA_PCT", "0.01"))
YOLO_MAX_AREA_PCT = float(os.getenv("YOLO_MAX_AREA_PCT", "0.8"))
YOLO_MIN_ASPECT = float(os.getenv("YOLO_MIN_ASPECT", "0.5"))
YOLO_MAX_ASPECT = float(os.getenv("YOLO_MAX_ASPECT", "2.0"))
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
yolo_detector = None
yolo_class_names: Optional[Dict[int, str]] = None
inference_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCES", "2")))
models_ready = threading.Event()
model_load_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in UPLOAD_EXTENSIONS


def load_models(load_detector: bool = True):
    """Load feature extractor, regressor, and optional YOLO detector."""
    global feature_extractor, regressor, feature_type, yolo_detector, yolo_class_names

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

    if not hasattr(regressor.scaler, "mean_"):
        vit_scaler_path = MODELS_DIR / "vit_scaler.joblib"
        if vit_scaler_path.exists():
            import joblib

            regressor.scaler = joblib.load(vit_scaler_path)
            logger.info("Loaded VIT scaler for %s", model_path.name)

    if load_detector and DETECTION_ENABLED and YOLO and YOLO_MODEL_PATH and YOLO_MODEL_PATH.exists():
        try:
            yolo_detector = YOLO(str(YOLO_MODEL_PATH))
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
            "color": "#ffff00",
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


def run_detection(image: Image.Image) -> List[Dict[str, float]]:
    """Return bbox detections if YOLO detector is configured."""
    if yolo_detector is None:
        return []
    try:
        results = yolo_detector(image, verbose=False)
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
        if class_name and class_name != "Crab":
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
    return filtered


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


async def predict_image(file: UploadFile) -> Dict[str, object]:
    if not models_ready.is_set():
        deadline = asyncio.get_running_loop().time() + float(os.getenv("MODEL_READY_TIMEOUT", "20"))
        while not models_ready.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.2)
        if not models_ready.is_set():
            raise HTTPException(status_code=503, detail="Model loading, please retry shortly.")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    raw_bytes = await file.read()
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    raw_bboxes = run_detection(image) if DETECTION_ENABLED else []
    bboxes = filter_bboxes(image, raw_bboxes) if DETECTION_ENABLED else []
    primary_bbox = select_primary_bbox(bboxes) if DETECTION_CROP_ENABLED else None
    roi_image = crop_to_bbox(image, primary_bbox) if primary_bbox else image
    estimate_input = "yolo_crop" if primary_bbox else "whole_image_fallback" if DETECTION_ENABLED else "whole_image"

    np_image = np.array(roi_image)
    features = feature_extractor.extract_features(np_image).reshape(1, -1)
    days_until_molt = float(regressor.predict(features)[0])
    estimated_molt_event_date = get_estimated_molt_event_date(days_until_molt)
    phase_info = get_molt_phase_category(days_until_molt)

    label_text = f"{phase_info['phase']} ({days_until_molt:.1f}d)"
    thumbnail = encode_thumbnail(image, label_text, bboxes, phase_info["color"])
    recommendation = phase_info["recommendation"]
    if DETECTION_ENABLED and not bboxes:
        recommendation = (
            "No confident crab detection. Estimate was run on the full image; "
            "try a closer, centered shot for a crop-based estimate."
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
        "bbox_count": len(bboxes),
        "crop_used": primary_bbox is not None,
        "primary_bbox": primary_bbox,
        "bboxes": bboxes,
        "image_width": image.width,
        "image_height": image.height,
        "detection_debug": {
            "enabled": DETECTION_ENABLED,
            "raw_count": len(raw_bboxes),
            "filtered_count": len(bboxes),
            "class_filter": "Crab",
            "filters": {
                "conf_min": YOLO_CONF_MIN,
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
        "yolo_detector": yolo_detector is not None,
        "models_ready": models_ready.is_set(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    async with inference_semaphore:
        try:
            result = await predict_image(file)
            return JSONResponse(content=result)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict_stream")
async def predict_stream(file: UploadFile = File(...)):
    async with inference_semaphore:
        try:
            result = await predict_image(file)
            return JSONResponse(content=result)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.exception("Streaming prediction failed")
            raise HTTPException(status_code=500, detail=f"Streaming prediction failed: {exc}") from exc


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
    return {"message": "Green Crab Molt Detector API (FastAPI)", "endpoints": ["/predict", "/predict_stream", "/health"]}


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
