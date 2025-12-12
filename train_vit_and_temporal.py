"""
Retrain ViT-based regression and temporal models using the manifest.

Outputs:
- models/vit_features.npy
- models/vit_metadata.json
- models/vit_scaler.joblib
- models/molt_regressor_vit_random_forest.joblib
- models/molt_regressor_vit_temporal.joblib
- reports/vit_metrics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from src.feature_extractor import GeneralCrustaceanFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["molt_date"] = pd.to_datetime(df["molt_date"])
    return df


def extract_vit_features(df: pd.DataFrame, device: str | None = None) -> Tuple[np.ndarray, pd.Series]:
    extractor = GeneralCrustaceanFeatureExtractor("vit_base", device=device)
    features: List[np.ndarray] = []
    labels: List[float] = []
    groups: List[str] = []
    idx_keep: List[int] = []

    labeled_df = df[df["days_until_molt"].notna()].copy()
    logger.info("Extracting features for %d labeled images", len(labeled_df))
    for idx, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc="ViT features"):
        feat = extractor.extract_features(row["image_path"])
        features.append(feat)
        labels.append(float(row["days_until_molt"]))
        groups.append(str(row["crab_id"]))
        idx_keep.append(idx)

    feature_matrix = np.stack(features)
    label_series = pd.Series(labels, index=idx_keep)
    group_series = pd.Series(groups, index=idx_keep)
    return feature_matrix, label_series, group_series


def train_regressor(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_path: Path, scaler_path: Path
) -> Dict[str, float]:
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"algorithm": "random_forest", "model": model, "scaler": scaler, "is_fitted": True},
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    logger.info("Saved regressor (with scaler) to %s", model_path)
    logger.info("MAE: %.3f | R2: %.3f", mae, r2)

    return {"mae": float(mae), "r2": float(r2), "n_train": len(train_idx), "n_test": len(test_idx)}


def build_temporal_sequences(
    df: pd.DataFrame, features: np.ndarray, feature_indices: List[int], window_size: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Map index -> row position in features
    feat_map = {idx: pos for pos, idx in enumerate(feature_indices)}
    sequences = []
    targets = []
    groups = []

    labeled_df = df.loc[feature_indices].copy()
    labeled_df["capture_date"] = pd.to_datetime(labeled_df["capture_date"])
    grouped = labeled_df.sort_values("capture_date").groupby("crab_id")

    for crab_id, crab_df in grouped:
        idxs = list(crab_df.index)
        if len(idxs) < window_size:
            continue
        for i in range(len(idxs) - window_size + 1):
            window_idxs = idxs[i : i + window_size]
            if any(pd.isna(labeled_df.loc[j, "days_until_molt"]) for j in window_idxs):
                continue
            feat_concat = np.concatenate([features[feat_map[j]] for j in window_idxs])
            target = float(labeled_df.loc[window_idxs[-1], "days_until_molt"])
            sequences.append(feat_concat)
            targets.append(target)
            groups.append(str(crab_id))

    if not sequences:
        raise ValueError("No temporal sequences constructed; check labeled data and window_size.")

    X_seq = np.stack(sequences)
    y_seq = np.array(targets)
    group_seq = np.array(groups)
    return X_seq, y_seq, group_seq


def train_temporal(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_path: Path, scaler_path: Path
) -> Dict[str, float]:
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(
        {"algorithm": "random_forest", "model": model, "scaler": scaler, "is_fitted": True},
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    logger.info("Saved temporal regressor (with scaler) to %s", model_path)
    logger.info("Temporal MAE: %.3f | R2: %.3f", mae, r2)

    return {"mae": float(mae), "r2": float(r2), "n_train": len(train_idx), "n_test": len(test_idx)}


def main():
    project_root = Path(__file__).parent
    manifest_path = project_root / "data" / "processed" / "manifest.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    df = load_manifest(manifest_path)

    # Extract features
    vit_features, labels, groups = extract_vit_features(df)
    feature_indices = list(labels.index)

    # Save raw features and metadata
    np.save(models_dir / "vit_features.npy", vit_features)
    metadata = {
        "num_samples": int(vit_features.shape[0]),
        "feature_dim": int(vit_features.shape[1]),
        "indices": feature_indices,
    }
    with (models_dir / "vit_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Train base regressor
    reg_metrics = train_regressor(
        vit_features, labels.values, np.array(groups.values), models_dir / "molt_regressor_vit_random_forest.joblib", models_dir / "vit_scaler.joblib"
    )

    # Temporal sequences
    X_seq, y_seq, groups_seq = build_temporal_sequences(df, vit_features, feature_indices, window_size=3)
    temporal_metrics = train_temporal(
        X_seq,
        y_seq,
        groups_seq,
        models_dir / "molt_regressor_vit_temporal.joblib",
        models_dir / "vit_temporal_scaler.joblib",
    )

    # Save metrics
    metrics = {"vit_regressor": reg_metrics, "vit_temporal": temporal_metrics}
    with (reports_dir / "vit_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", reports_dir / "vit_metrics.json")


if __name__ == "__main__":
    main()
