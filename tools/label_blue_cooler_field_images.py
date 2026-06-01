#!/usr/bin/env python3
"""
Create a draft field-label spreadsheet for the May 29 blue cooler images.

The script uses only models available in this repo/environment:
- YOLO detector for crab boxes
- ViT feature extractor + saved molt regressor for days-to-molt estimates
- A lightweight orientation classifier trained from data/sam3_orientation

There is no trained sex classifier in this repo. Sex is filled from the field note
that the blue cooler crabs were male, and provenance columns make that explicit.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_extractor import GeneralCrustaceanFeatureExtractor  # noqa: E402
from model import MoltPhaseRegressor  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "data" / "raw" / "Green Crab AI 2026"
OUTPUT_CSV = REPO_ROOT / "field_data" / "blue_cooler_may29_estimated_labels.csv"
ORIENTATION_DIR = REPO_ROOT / "data" / "sam3_orientation"
YOLO_MODEL_PATH = REPO_ROOT / "models" / "fathomnet_mvp_yolov8_1280_20240914.pt"
MOLT_MODEL_PATH = REPO_ROOT / "models" / "best_vit_regressor.joblib"
VIT_SCALER_PATH = REPO_ROOT / "models" / "vit_scaler.joblib"
CAPTURE_DATE = date(2026, 5, 29)

YOLO_CONF_MIN = 0.20
YOLO_MIN_AREA_PCT = 0.01
YOLO_MAX_AREA_PCT = 0.85
YOLO_MIN_ASPECT = 0.35
YOLO_MAX_ASPECT = 3.25
ORIENTATION_CONF_MIN = 0.55


@dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: Optional[int]
    class_name: Optional[str]

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1.0)


def image_paths(input_dir: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in extensions)


def phase_from_days(days_until_molt: float) -> str:
    if days_until_molt < 0:
        return "molted"
    if days_until_molt <= 3:
        return "peeler_imminent"
    if days_until_molt <= 14:
        return "pre_molt"
    return "intermolt"


def estimated_molt_event_date(days_until_molt: float) -> str:
    return (CAPTURE_DATE + timedelta(days=float(days_until_molt))).isoformat()


def app_confidence_from_days(days_until_molt: float) -> str:
    return "high" if abs(days_until_molt) < 20 else "medium"


def allowed_detection(image: Image.Image, det: Detection) -> bool:
    width, height = image.size
    area_pct = (det.width * det.height) / max(width * height, 1)
    if det.class_name and det.class_name != "Crab":
        return False
    if det.confidence < YOLO_CONF_MIN:
        return False
    if area_pct < YOLO_MIN_AREA_PCT or area_pct > YOLO_MAX_AREA_PCT:
        return False
    if det.aspect < YOLO_MIN_ASPECT or det.aspect > YOLO_MAX_ASPECT:
        return False
    return True


def run_detection(detector: YOLO, image: Image.Image) -> Tuple[List[Detection], List[Detection]]:
    results = detector(image, verbose=False)
    raw: List[Detection] = []
    names = getattr(getattr(detector, "model", None), "names", None)
    if results and getattr(results[0], "boxes", None) is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            cls = int(box.cls[0]) if box.cls is not None else None
            class_name = names.get(cls) if isinstance(names, dict) and cls in names else None
            raw.append(
                Detection(
                    xmin=float(xyxy[0]),
                    ymin=float(xyxy[1]),
                    xmax=float(xyxy[2]),
                    ymax=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                    class_name=class_name,
                )
            )
    filtered = [det for det in raw if allowed_detection(image, det)]
    return raw, filtered


def select_primary_detection(detections: Sequence[Detection], image: Image.Image) -> Optional[Detection]:
    if not detections:
        return None
    width, height = image.size
    image_area = max(width * height, 1)

    def score(det: Detection) -> float:
        area_pct = (det.width * det.height) / image_area
        # Favor confident, large-enough detections without letting very large boxes dominate.
        return det.confidence + min(area_pct, 0.45)

    return max(detections, key=score)


def crop_detection(image: Image.Image, det: Detection, padding_pct: float = 0.08) -> Image.Image:
    width, height = image.size
    pad_x = int(det.width * padding_pct)
    pad_y = int(det.height * padding_pct)
    xmin = max(0, int(det.xmin) - pad_x)
    ymin = max(0, int(det.ymin) - pad_y)
    xmax = min(width, int(det.xmax) + pad_x)
    ymax = min(height, int(det.ymax) + pad_y)
    if xmax <= xmin or ymax <= ymin:
        return image
    return image.crop((xmin, ymin, xmax, ymax))


def estimate_distance(image: Image.Image, det: Optional[Detection]) -> str:
    if det is None:
        return "unknown"
    width, height = image.size
    area_pct = (det.width * det.height) / max(width * height, 1)
    if area_pct >= 0.35:
        return "close_under_6in"
    if area_pct >= 0.08:
        return "medium_6_12in"
    return "far_over_12in"


def estimate_lighting(image: Image.Image) -> str:
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 70:
        return "low_light"
    if std > 70:
        return "mixed"
    if mean > 165:
        return "sun"
    return "shade"


def estimate_position(image: Image.Image, det: Optional[Detection]) -> str:
    if det is None:
        return "unknown"
    width, height = image.size
    margin_x = 0.03 * width
    margin_y = 0.03 * height
    if det.xmin <= margin_x or det.ymin <= margin_y or det.xmax >= width - margin_x or det.ymax >= height - margin_y:
        return "partially_cut_off"
    cx = (det.xmin + det.xmax) / 2
    cy = (det.ymin + det.ymax) / 2
    if abs(cx - width / 2) <= 0.25 * width and abs(cy - height / 2) <= 0.25 * height:
        return "centered"
    return "edge"


def estimate_motion_blur(image: Image.Image) -> str:
    # Simple edge-energy proxy using adjacent pixel differences; avoids requiring cv2.
    gray = np.array(image.convert("L").resize((256, 256)), dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    edge_energy = float((dx + dy) / 2)
    if edge_energy < 4.0:
        return "severe"
    if edge_energy < 7.0:
        return "slight"
    return "none"


def extract_feature(extractor: GeneralCrustaceanFeatureExtractor, image: Image.Image) -> np.ndarray:
    return extractor.extract_features(np.array(image.convert("RGB"))).reshape(1, -1)


def load_molt_regressor() -> MoltPhaseRegressor:
    regressor = MoltPhaseRegressor("random_forest")
    regressor.load_model(MOLT_MODEL_PATH)
    if not hasattr(regressor.scaler, "mean_") and VIT_SCALER_PATH.exists():
        regressor.scaler = joblib.load(VIT_SCALER_PATH)
    return regressor


def orientation_training_paths() -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []
    for label in ("ventral", "dorsal", "uncertain"):
        label_dir = ORIENTATION_DIR / label
        if not label_dir.exists():
            continue
        for path in image_paths(label_dir):
            rows.append((path, label))
    return rows


def train_orientation_classifier(
    extractor: GeneralCrustaceanFeatureExtractor,
) -> Optional[LogisticRegression]:
    rows = orientation_training_paths()
    if len(rows) < 20:
        logger.warning("Not enough orientation examples; orientation estimates will be unknown.")
        return None
    features: List[np.ndarray] = []
    labels: List[str] = []
    for path, label in rows:
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                features.append(extract_feature(extractor, img).ravel())
                labels.append(label)
        except Exception as exc:
            logger.warning("Skipping orientation training image %s: %s", path, exc)
    if len(set(labels)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(np.vstack(features), labels)
    logger.info("Trained orientation classifier from %d examples: %s", len(labels), sorted(set(labels)))
    return clf


def estimate_orientation(
    classifier: Optional[LogisticRegression],
    extractor: GeneralCrustaceanFeatureExtractor,
    crop: Image.Image,
    det: Optional[Detection],
) -> Tuple[str, float, str]:
    if classifier is None:
        return "unknown", 0.0, "not_estimated_no_orientation_classifier"
    features = extract_feature(extractor, crop)
    probabilities = classifier.predict_proba(features)[0]
    classes = list(classifier.classes_)
    best_idx = int(np.argmax(probabilities))
    label = str(classes[best_idx])
    confidence = float(probabilities[best_idx])

    if confidence < ORIENTATION_CONF_MIN or label == "uncertain":
        # The repo has no side-view class. Mark high-aspect uncertain images as side,
        # which should be reviewed by a human because this is only a weak heuristic.
        if det is not None and det.aspect >= 2.0:
            return "side", confidence, "heuristic_from_uncertain_orientation_and_bbox_aspect"
        return "unknown", confidence, "orientation_classifier_low_confidence_or_uncertain"
    return label, confidence, "vit_logistic_regression_trained_on_data_sam3_orientation"


def row_for_image(
    path: Path,
    detector: YOLO,
    feature_extractor: GeneralCrustaceanFeatureExtractor,
    molt_regressor: MoltPhaseRegressor,
    orientation_classifier: Optional[LogisticRegression],
) -> Dict[str, object]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    raw_dets, filtered_dets = run_detection(detector, image)
    primary = select_primary_detection(filtered_dets, image)
    crop = crop_detection(image, primary) if primary else image

    whole_features = extract_feature(feature_extractor, image)
    whole_days_float = float(molt_regressor.predict(whole_features)[0])
    whole_days = round(whole_days_float, 3)
    whole_event_date = estimated_molt_event_date(whole_days_float)
    whole_phase = phase_from_days(whole_days_float)

    crop_days = ""
    crop_event_date = ""
    crop_phase = ""
    app_estimate_input = "whole_image_fallback"
    app_days = whole_days
    app_event_date = whole_event_date
    app_phase = whole_phase
    app_confidence = app_confidence_from_days(whole_days_float)
    molt_model = "best_vit_regressor.joblib_on_whole_image_fallback"
    if primary is not None:
        features = extract_feature(feature_extractor, crop)
        crop_days_float = float(molt_regressor.predict(features)[0])
        crop_days = round(crop_days_float, 3)
        crop_event_date = estimated_molt_event_date(crop_days_float)
        crop_phase = phase_from_days(crop_days_float)
        app_estimate_input = "yolo_crop"
        app_days = crop_days
        app_event_date = crop_event_date
        app_phase = crop_phase
        app_confidence = app_confidence_from_days(crop_days_float)
        molt_model = "best_vit_regressor.joblib_on_yolo_primary_crop"

    view_angle, view_confidence, view_source = estimate_orientation(
        orientation_classifier, feature_extractor, crop, primary
    )

    image_id = f"2026-05-29_blue_cooler_{path.stem}"
    notes = []
    if primary is None:
        notes.append("No filtered YOLO crab detection; selected app estimate uses whole image fallback.")
    if len(filtered_dets) > 1:
        notes.append("Multiple filtered crab detections; primary detection selected automatically.")
    if view_angle in {"unknown", "side"}:
        notes.append("View angle is provisional; repo has no side-view classifier.")

    bbox_area_pct = ""
    bbox_aspect = ""
    bbox_conf = ""
    if primary is not None:
        width, height = image.size
        bbox_area_pct = round((primary.width * primary.height) / max(width * height, 1), 5)
        bbox_aspect = round(primary.aspect, 3)
        bbox_conf = round(primary.confidence, 4)

    return {
        "date": "2026-05-29",
        "condo_id": "blue_cooler",
        "image_id": image_id,
        "icloud_album_name": "local:data/raw/Green Crab AI 2026",
        "image_filename": path.name,
        "photographer": "Grace Welles; Gabriela Bradt",
        "view_angle": view_angle,
        "distance_category": estimate_distance(image, primary),
        "lighting": estimate_lighting(image),
        "background": "hand",
        "crab_in_frame_count": "multiple" if len(filtered_dets) > 1 else "1" if primary is not None else "partial",
        "crab_position": estimate_position(image, primary),
        "motion_blur": estimate_motion_blur(crop),
        "sex": "male",
        "known_molt_phase": "unknown",
        "days_until_molt_if_known": "",
        "molt_event_date": "",
        "app_estimated_days_to_molt": app_days,
        "app_estimated_molt_event_date": app_event_date,
        "app_phase": app_phase,
        "app_confidence": app_confidence,
        "app_estimate_input": app_estimate_input,
        "whole_image_estimated_days_to_molt": whole_days,
        "whole_image_estimated_molt_event_date": whole_event_date,
        "whole_image_app_phase": whole_phase,
        "yolo_crop_estimated_days_to_molt": crop_days,
        "yolo_crop_estimated_molt_event_date": crop_event_date,
        "yolo_crop_app_phase": crop_phase,
        "human_confidence": "unknown",
        "shell_condition_notes": "",
        "limb_loss_or_injury": "",
        "notes": " ".join(notes),
        "review_status": "new",
        "view_angle_label_source": view_source,
        "view_angle_model": "ViT-B/16 ImageNet features + LogisticRegression on data/sam3_orientation; side is a bbox-aspect heuristic",
        "view_angle_confidence": round(view_confidence, 4),
        "sex_label_source": "user_field_note_blue_cooler_all_male; no sex classifier found in repo",
        "sex_model": "not_estimated",
        "app_molt_model": molt_model,
        "detector_model": str(YOLO_MODEL_PATH.relative_to(REPO_ROOT)),
        "raw_detection_count": len(raw_dets),
        "filtered_detection_count": len(filtered_dets),
        "primary_bbox_confidence": bbox_conf,
        "primary_bbox_area_pct": bbox_area_pct,
        "primary_bbox_aspect": bbox_aspect,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paths = image_paths(INPUT_DIR)
    if not paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")
    logger.info("Found %d field images in %s", len(paths), INPUT_DIR)

    logger.info("Loading YOLO detector from %s", YOLO_MODEL_PATH)
    detector = YOLO(str(YOLO_MODEL_PATH))

    logger.info("Loading ViT feature extractor")
    feature_extractor = GeneralCrustaceanFeatureExtractor("vit_base")

    logger.info("Loading molt regressor from %s", MOLT_MODEL_PATH)
    molt_regressor = load_molt_regressor()

    orientation_classifier = train_orientation_classifier(feature_extractor)

    rows: List[Dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        logger.info("[%d/%d] Labeling %s", index, len(paths), path.name)
        try:
            rows.append(row_for_image(path, detector, feature_extractor, molt_regressor, orientation_classifier))
        except Exception as exc:
            logger.exception("Failed to label %s", path)
            rows.append(
                {
                    "date": "2026-05-29",
                    "condo_id": "blue_cooler",
                    "image_id": f"2026-05-29_blue_cooler_{path.stem}",
                    "icloud_album_name": "local:data/raw/Green Crab AI 2026",
                    "image_filename": path.name,
                    "photographer": "Grace Welles; Gabriela Bradt",
                    "view_angle": "unknown",
                    "distance_category": "unknown",
                    "lighting": "unknown",
                    "background": "hand",
                    "crab_in_frame_count": "partial",
                    "crab_position": "unknown",
                    "motion_blur": "unknown",
                    "sex": "male",
                    "known_molt_phase": "unknown",
                    "days_until_molt_if_known": "",
                    "molt_event_date": "",
                    "app_estimated_days_to_molt": "",
                    "app_estimated_molt_event_date": "",
                    "app_phase": "unknown",
                    "app_confidence": "low",
                    "app_estimate_input": "not_run",
                    "whole_image_estimated_days_to_molt": "",
                    "whole_image_estimated_molt_event_date": "",
                    "whole_image_app_phase": "",
                    "yolo_crop_estimated_days_to_molt": "",
                    "yolo_crop_estimated_molt_event_date": "",
                    "yolo_crop_app_phase": "",
                    "human_confidence": "unknown",
                    "shell_condition_notes": "",
                    "limb_loss_or_injury": "",
                    "notes": f"Automated labeling failed: {exc}",
                    "review_status": "new",
                    "view_angle_label_source": "not_estimated_error",
                    "view_angle_model": "not_estimated",
                    "view_angle_confidence": "",
                    "sex_label_source": "user_field_note_blue_cooler_all_male; no sex classifier found in repo",
                    "sex_model": "not_estimated",
                    "app_molt_model": "not_run_error",
                    "detector_model": str(YOLO_MODEL_PATH.relative_to(REPO_ROOT)),
                    "raw_detection_count": "",
                    "filtered_detection_count": "",
                    "primary_bbox_confidence": "",
                    "primary_bbox_area_pct": "",
                    "primary_bbox_aspect": "",
                }
            )

    write_csv(OUTPUT_CSV, rows)
    logger.info("Wrote %d rows to %s", len(rows), OUTPUT_CSV)


if __name__ == "__main__":
    main()
