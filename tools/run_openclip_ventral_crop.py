#!/usr/bin/env python3
"""
OpenCLIP feature extraction focusing on ventral plate crops + regressors and t-SNE.

Outputs:
- models/openclip_ventral_features.npy
- models/openclip_ventral_metadata.json
- models/openclip_ventral_regressor.joblib (+ scaler)
- models/openclip_ventral_temporal_w5.joblib (+ scaler)
- reports/openclip_ventral_metrics.json
- plots/openclip_ventral_tsne_molt_phases.png
"""

from __future__ import annotations

import json
import logging
import random
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df = df[df["days_until_molt"].notna()].copy()
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["crab_id"] = df["crab_id"].astype(str)
    df["phase"] = df["days_until_molt"].astype(float).apply(phase_from_days)
    return df


def ventral_crops(image: Image.Image, n_crops: int = 3) -> List[Image.Image]:
    """Generate ventral-biased crops with light jitter."""
    w, h = image.size
    crops = []
    for _ in range(n_crops):
        # Focus on lower-central region; jitter up to 5% of width/height
        jitter_x = int(0.05 * w)
        jitter_y = int(0.05 * h)
        left = int(0.1 * w + random.randint(-jitter_x, jitter_x))
        right = int(0.9 * w + random.randint(-jitter_x, jitter_x))
        top = int(0.45 * h + random.randint(-jitter_y, jitter_y))
        bottom = h  # always include lower portion
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        top = max(0, min(top, h - 2))
        bottom = max(top + 1, min(bottom, h))
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def extract_features(df: pd.DataFrame, device: str = "cpu", n_crops: int = 3) -> Tuple[np.ndarray, List[int]]:
    logger.info("Loading OpenCLIP (ViT-H/14, laion2B-s32B-b79K) on %s", device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model = model.to(device)
    model.eval()

    feats: List[np.ndarray] = []
    idxs: List[int] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ventral OpenCLIP feats"):
        img = Image.open(row["image_path"]).convert("RGB")
        crops = ventral_crops(img, n_crops=n_crops)
        crop_feats = []
        with torch.no_grad():
            for crop in crops:
                tensor = preprocess(crop).unsqueeze(0).to(device)
                f = model.encode_image(tensor)
                f = f / f.norm(dim=-1, keepdim=True)
                crop_feats.append(f.cpu().numpy().squeeze().astype(np.float32))
        feat = np.stack(crop_feats).mean(axis=0)
        feats.append(feat)
        idxs.append(idx)
    return np.stack(feats), idxs


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
        n_estimators=800, min_samples_split=4, min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return {"mae": float(mae), "r2": float(r2), "n_train": len(train_idx), "n_test": len(test_idx)}


def build_sequences(
    df: pd.DataFrame, feats: np.ndarray, feat_indices: List[int], window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = {idx: i for i, idx in enumerate(feat_indices)}
    seqs, targets, groups = [], [], []
    labeled_df = df.loc[feat_indices].copy()
    labeled_df["capture_date"] = pd.to_datetime(labeled_df["capture_date"])
    for crab_id, cdf in labeled_df.groupby("crab_id"):
        cdf = cdf.sort_values("capture_date")
        idxs = list(cdf.index)
        if len(idxs) < window:
            continue
        for i in range(len(idxs) - window + 1):
            win = idxs[i : i + window]
            if cdf.loc[win, "days_until_molt"].isna().any():
                continue
            seqs.append(np.concatenate([feats[pos[j]] for j in win]))
            targets.append(float(cdf.loc[win[-1], "days_until_molt"]))
            groups.append(str(crab_id))
    if not seqs:
        raise ValueError("No sequences built; window too large.")
    return np.stack(seqs), np.array(targets), np.array(groups)


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


def plot_tsne(feats: np.ndarray, df: pd.DataFrame, idxs: List[int], out_path: Path):
    df_sub = df.loc[idxs].copy()
    phases = df_sub["phase"].tolist()
    tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1), random_state=42, init="pca")
    emb = tsne.fit_transform(feats)
    cmap = {"imminent": "#d62728", "near": "#ff7f0e", "mid": "#1f77b4", "far": "#2ca02c"}
    plt.figure(figsize=(10, 8))
    for phase in sorted(set(phases)):
        mask = [p == phase for p in phases]
        plt.scatter(emb[mask, 0], emb[mask, 1], s=28, alpha=0.8, label=phase, c=cmap.get(phase, "#7f7f7f"))
    plt.legend(title="Molt phase")
    plt.title("OpenCLIP ventral-crop t-SNE (labeled)")
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved t-SNE to %s", out_path)


def main():
    random.seed(42)
    np.random.seed(42)
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "processed" / "manifest_with_2016_labels.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    plots_dir = project_root / "plots"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    df = load_manifest(manifest_path)
    logger.info("Labeled rows: %d, crabs: %d", len(df), df["crab_id"].nunique())

    # Extract (or cache) ventral features
    feat_file = models_dir / "openclip_ventral_features.npy"
    meta_file = models_dir / "openclip_ventral_metadata.json"
    if feat_file.exists() and meta_file.exists():
        logger.info("Loading cached ventral features from %s", feat_file)
        feats = np.load(feat_file)
        meta = json.loads(meta_file.read_text())
        idxs = meta["indices"]
    else:
        feats, idxs = extract_features(df, device="cpu", n_crops=3)
        np.save(feat_file, feats)
        meta = {"num_samples": int(feats.shape[0]), "feature_dim": int(feats.shape[1]), "indices": idxs}
        meta_file.write_text(json.dumps(meta, indent=2))

    labels = df.loc[idxs, "days_until_molt"].astype(float).values
    groups = df.loc[idxs, "crab_id"].astype(str).values

    single_metrics = train_regressor(
        feats,
        labels,
        groups,
        models_dir / "openclip_ventral_regressor.joblib",
        models_dir / "openclip_ventral_scaler.joblib",
    )
    logger.info("Ventral single-image MAE %.3f R2 %.3f", single_metrics["mae"], single_metrics["r2"])

    window = 5
    X_seq, y_seq, g_seq = build_sequences(df, feats, idxs, window=window)
    temporal_metrics = train_temporal(
        X_seq,
        y_seq,
        g_seq,
        models_dir / f"openclip_ventral_temporal_w{window}.joblib",
        models_dir / f"openclip_ventral_temporal_scaler_w{window}.joblib",
    )
    logger.info("Ventral temporal (w=%d) MAE %.3f R2 %.3f", window, temporal_metrics["mae"], temporal_metrics["r2"])

    metrics = {"single_image": single_metrics, "temporal_w5": temporal_metrics, "meta": meta}
    (reports_dir / "openclip_ventral_metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_tsne(feats, df, idxs, plots_dir / "openclip_ventral_tsne_molt_phases.png")


if __name__ == "__main__":
    main()
