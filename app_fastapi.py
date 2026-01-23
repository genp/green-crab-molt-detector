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
from fastapi.responses import HTMLResponse, JSONResponse
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
inference_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCES", "2")))


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in UPLOAD_EXTENSIONS


def load_models():
    """Load feature extractor, regressor, and optional YOLO detector."""
    global feature_extractor, regressor, feature_type, yolo_detector

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

    if DETECTION_ENABLED and YOLO and YOLO_MODEL_PATH and YOLO_MODEL_PATH.exists():
        try:
            yolo_detector = YOLO(str(YOLO_MODEL_PATH))
            logger.info("Loaded YOLO detector for bboxes from %s", YOLO_MODEL_PATH)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Failed to load YOLO detector: %s", exc)


def get_molt_phase_category(days_until_molt: float) -> Dict[str, object]:
    """Convert days until molt to category and recommendation."""
    if days_until_molt < 0:
        return {
            "phase": "Post-molt",
            "color": "success",
            "recommendation": "Crab has recently molted. Shell is likely soft.",
            "harvest_ready": False,
        }
    if days_until_molt <= 3:
        return {
            "phase": "Peeler (Imminent molt)",
            "color": "danger",
            "recommendation": "HARVEST NOW! Crab will molt within 3 days.",
            "harvest_ready": True,
        }
    if days_until_molt <= 7:
        return {
            "phase": "Pre-molt (Near)",
            "color": "warning",
            "recommendation": "Monitor closely. Harvest window approaching.",
            "harvest_ready": False,
        }
    if days_until_molt <= 14:
        return {
            "phase": "Pre-molt (Early)",
            "color": "info",
            "recommendation": "Check again in a week.",
            "harvest_ready": False,
        }
    return {
        "phase": "Inter-molt",
        "color": "primary",
        "recommendation": "Crab is not close to molting.",
        "harvest_ready": False,
    }


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
                boxes_out.append(
                    {
                        "xmin": xyxy[0],
                        "ymin": xyxy[1],
                        "xmax": xyxy[2],
                        "ymax": xyxy[3],
                        "confidence": conf,
                        "class": cls,
                    }
                )
        return boxes_out
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Detection failed: %s", exc)
        return []


def encode_thumbnail(image: Image.Image, label: str, bboxes: List[Dict[str, float]]) -> str:
    """Create a small JPEG thumbnail with overlay text and optional bbox outlines."""
    thumb = image.copy().convert("RGB")
    orig_w, orig_h = thumb.size
    thumb.thumbnail((320, 320))
    draw = ImageDraw.Draw(thumb)

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
                outline="red",
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
    if feature_extractor is None or regressor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    raw_bytes = await file.read()
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    np_image = np.array(image)
    features = feature_extractor.extract_features(np_image).reshape(1, -1)
    days_until_molt = float(regressor.predict(features)[0])
    phase_info = get_molt_phase_category(days_until_molt)

    bboxes = run_detection(image)
    label_text = f"{phase_info['phase']} ({days_until_molt:.1f}d)"
    thumbnail = encode_thumbnail(image, label_text, bboxes)

    return {
        "success": True,
        "days_until_molt": days_until_molt,
        "phase": phase_info["phase"],
        "color": phase_info["color"],
        "recommendation": phase_info["recommendation"],
        "harvest_ready": phase_info["harvest_ready"],
        "confidence": "High" if abs(days_until_molt) < 20 else "Medium",
        "thumbnail": thumbnail,
        "feature_type": feature_type.upper() if feature_type else None,
        "bboxes": bboxes,
    }


@app.on_event("startup")
def startup_event():
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


@app.get("/")
def root():
    return {"message": "Green Crab Molt Detector API (FastAPI)", "endpoints": ["/predict", "/predict_stream", "/health"]}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
