#!/usr/bin/env python3
"""
Extract OpenCLIP image features and train molt regressors (single-image + temporal).

Outputs:
- models/openclip_features.npy
- models/openclip_metadata.json
- models/openclip_regressor.joblib (single-image)
- models/openclip_regressor_scaler.joblib
- models/openclip_temporal_w5.joblib (temporal window=5)
- models/openclip_temporal_scaler_w5.joblib
- reports/openclip_metrics.json
- plots/openclip_tsne_molt_phases.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def phase_from_days(days_until_molt: float) -> str:
    if days_until_molt <= 3:
        return "imminent"
    if days_until_molt <= 7:
        return "near"
    if days_until_molt <= 14:
        return "mid"
    return "far"


def load_labeled_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df = df[df["days_until_molt"].notna()].copy()
    df["crab_id"] = df["crab_id"].astype(str)
    df["phase"] = df["days_until_molt"].astype(float).apply(phase_from_days)
    return df


def extract_openclip_features(df: pd.DataFrame, device: str = "cpu") -> Tuple[np.ndarray, List[int]]:
    logger.info("Loading OpenCLIP (ViT-H/14, laion2B-s32B-b79K) on %s", device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model = model.to(device)
    model.eval()

    features: List[np.ndarray] = []
    indices: List[int] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OpenCLIP features"):
        image_path = Path(row["image_path"])
        with torch.no_grad():
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy().squeeze().astype(np.float32))
            indices.append(idx)
    feature_matrix = np.stack(features)
    return feature_matrix, indices


def train_single_image(
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
        n_estimators=800, min_samples_split=4, min_samples_leaf=1, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return {"mae": float(mae), "r2": float(r2), "n_train": len(train_idx), "n_test": len(test_idx)}


def build_temporal_sequences(
    df: pd.DataFrame, features: np.ndarray, feature_indices: List[int], window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = {idx: i for i, idx in enumerate(feature_indices)}
    sequences = []
    targets = []
    groups = []

    labeled_df = df.loc[feature_indices].copy()
    labeled_df["capture_date"] = pd.to_datetime(labeled_df["capture_date"])
    for crab_id, crab_df in labeled_df.groupby("crab_id"):
        crab_df = crab_df.sort_values("capture_date")
        idxs = list(crab_df.index)
        if len(idxs) < window:
            continue
        for i in range(len(idxs) - window + 1):
            win = idxs[i : i + window]
            if crab_df.loc[win, "days_until_molt"].isna().any():
                continue
            seq_feat = np.concatenate([features[pos[j]] for j in win])
            sequences.append(seq_feat)
            targets.append(float(crab_df.loc[win[-1], "days_until_molt"]))
            groups.append(str(crab_id))

    if not sequences:
        raise ValueError("No temporal sequences built; window too large for dataset.")

    return np.stack(sequences), np.array(targets), np.array(groups)


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

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return {"mae": float(mae), "r2": float(r2), "n_train": len(train_idx), "n_test": len(test_idx)}


def plot_tsne(features: np.ndarray, df: pd.DataFrame, indices: List[int], out_path: Path):
    df_subset = df.loc[indices].copy()
    phases = df_subset["phase"].tolist()
    crabs = df_subset["crab_id"].tolist()

    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42, init="pca")
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    cmap = {"imminent": "#d62728", "near": "#ff7f0e", "mid": "#1f77b4", "far": "#2ca02c"}
    for phase in sorted(set(phases)):
        mask = [p == phase for p in phases]
        plt.scatter(emb[mask, 0], emb[mask, 1], s=28, alpha=0.8, label=phase, c=cmap.get(phase, "#7f7f7f"))
    plt.legend(title="Molt phase")
    plt.title("OpenCLIP features t-SNE (labeled images)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved t-SNE plot to %s", out_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "processed" / "manifest_with_2016_labels.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    plots_dir = project_root / "plots"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    device = "cpu"
    df = load_labeled_manifest(manifest_path)
    logger.info("Labeled samples: %d across %d crabs", len(df), df["crab_id"].nunique())

    # Feature extraction (reuse cache if present)
    feat_file = models_dir / "openclip_features.npy"
    meta_file = models_dir / "openclip_metadata.json"
    if feat_file.exists() and meta_file.exists():
        logger.info("Loading cached OpenCLIP features from %s", feat_file)
        openclip_features = np.load(feat_file)
        meta = json.loads(meta_file.read_text())
        feature_indices = meta["indices"]
    else:
        openclip_features, feature_indices = extract_openclip_features(df, device=device)
        np.save(feat_file, openclip_features)
        meta = {
            "num_samples": int(openclip_features.shape[0]),
            "feature_dim": int(openclip_features.shape[1]),
            "indices": feature_indices,
            "model": "openclip-ViT-H-14 laion2b_s32b_b79k",
        }
        meta_file.write_text(json.dumps(meta, indent=2))

    # Single-image regressor
    labels = df.loc[feature_indices, "days_until_molt"].astype(float).values
    groups = df.loc[feature_indices, "crab_id"].astype(str).values
    single_metrics = train_single_image(
        openclip_features,
        labels,
        groups,
        models_dir / "openclip_regressor.joblib",
        models_dir / "openclip_regressor_scaler.joblib",
    )
    logger.info("Single-image MAE %.3f R2 %.3f", single_metrics["mae"], single_metrics["r2"])

    # Temporal regressor (window=5)
    window = 5
    X_seq, y_seq, g_seq = build_temporal_sequences(df, openclip_features, feature_indices, window=window)
    temporal_metrics = train_temporal(
        X_seq,
        y_seq,
        g_seq,
        models_dir / f"openclip_temporal_w{window}.joblib",
        models_dir / f"openclip_temporal_scaler_w{window}.joblib",
    )
    logger.info("Temporal (w=%d) MAE %.3f R2 %.3f", window, temporal_metrics["mae"], temporal_metrics["r2"])

    # Save metrics
    metrics = {"single_image": single_metrics, "temporal_w5": temporal_metrics, "meta": meta}
    (reports_dir / "openclip_metrics.json").write_text(json.dumps(metrics, indent=2))

    # t-SNE visualization
    plot_tsne(openclip_features, df, feature_indices, plots_dir / "openclip_tsne_molt_phases.png")


if __name__ == "__main__":
    main()
