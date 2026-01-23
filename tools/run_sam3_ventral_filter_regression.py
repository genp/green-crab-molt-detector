#!/usr/bin/env python3
"""
Filter SAM3 ventral crops to ventral-only using OpenCLIP text prompts,
then retrain regressors and visualize t-SNE on the filtered subset.

This reuses:
- models/sam3_ventral_features.npy
- models/sam3_ventral_metadata.json
- data/processed/manifest_with_2016_labels.csv

Outputs:
- models/sam3_ventral_filtered_regressor.joblib (+ scaler)
- models/sam3_ventral_filtered_temporal_w5.joblib (+ scaler)  [if enough sequences]
- models/sam3_ventral_filtered_metadata.json  (decision + indices)
- reports/sam3_ventral_filtered_metrics.json
- plots/sam3_ventral_filtered_tsne.png
- plots/sam3_ventral_filtered_gallery.png
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
import torch
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


VENTRAL_PROMPT = "close-up photo of crab underside ventral plates, abdominal flap, high detail"
DORSAL_PROMPT = "photo of crab from above, dorsal shell and legs visible"


def load_data(root: Path):
    feats = np.load(root / "models" / "sam3_ventral_features.npy")
    meta = json.loads((root / "models" / "sam3_ventral_metadata.json").read_text())
    df = pd.read_csv(root / "data" / "processed" / "manifest_with_2016_labels.csv")
    df = df[df["days_until_molt"].notna()].copy()
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["crab_id"] = df["crab_id"].astype(str)
    return feats, meta, df


def score_ventral(feats: np.ndarray, device: str = "cpu", margin: float = 0.05):
    """
    Use OpenCLIP text embeddings to score each feature as ventral/dorsal/uncertain.
    Returns:
        labels: np.ndarray[int]  (1 = ventral, 0 = dorsal, -1 = uncertain)
        scores: list[dict] with raw sims and margin.
    """
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()

    texts = [VENTRAL_PROMPT, DORSAL_PROMPT]
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    vent_vec = text_emb[0:1]
    dor_vec = text_emb[1:2]

    labels = []
    scores_meta: List[Dict] = []
    # feats are already L2-normalized in the SAM3 extraction script.
    feat_t = torch.from_numpy(feats).to(device)
    with torch.no_grad():
        s_vent = (feat_t @ vent_vec.T).squeeze(-1)
        s_dor = (feat_t @ dor_vec.T).squeeze(-1)
        diff = (s_vent - s_dor).cpu().numpy()
        s_vent_np = s_vent.cpu().numpy()
        s_dor_np = s_dor.cpu().numpy()

    for i in range(len(diff)):
        d = float(diff[i])
        if d >= margin:
            lab = 1
        elif d <= -margin:
            lab = 0
        else:
            lab = -1
        labels.append(lab)
        scores_meta.append(
            {
                "s_ventral": float(s_vent_np[i]),
                "s_dorsal": float(s_dor_np[i]),
                "margin": d,
                "label": int(lab),
            }
        )

    labels_arr = np.array(labels, dtype=int)
    logger.info(
        "Orientation counts: ventral=%d, dorsal=%d, uncertain=%d",
        int((labels_arr == 1).sum()),
        int((labels_arr == 0).sum()),
        int((labels_arr == -1).sum()),
    )
    return labels_arr, scores_meta


def train_single(X, y, g):
    split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr, te = next(split.split(X, y, g))
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xte_s = scaler.transform(X[te])
    model = RandomForestRegressor(
        n_estimators=900, min_samples_split=4, min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(Xtr_s, y[tr])
    preds = model.predict(Xte_s)
    return model, scaler, {
        "mae": float(mean_absolute_error(y[te], preds)),
        "r2": float(r2_score(y[te], preds)),
        "n_train": len(tr),
        "n_test": len(te),
    }


def build_sequences(df, feats, idxs, window):
    pos = {idx: i for i, idx in enumerate(idxs)}
    seqs, targs, groups = [], [], []
    df = df.loc[idxs].copy()
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    for crab_id, cdf in df.groupby("crab_id"):
        cdf = cdf.sort_values("capture_date")
        idx_list = list(cdf.index)
        if len(idx_list) < window:
            continue
        for i in range(len(idx_list) - window + 1):
            win = idx_list[i : i + window]
            if cdf.loc[win, "days_until_molt"].isna().any():
                continue
            seqs.append(np.concatenate([feats[pos[j]] for j in win]))
            targs.append(float(cdf.loc[win[-1], "days_until_molt"]))
            groups.append(str(crab_id))
    if not seqs:
        feat_dim = feats.shape[1]
        return (
            np.empty((0, feat_dim * window), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    return np.stack(seqs), np.array(targs), np.array(groups)


def train_temporal(X, y, g):
    split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr, te = next(split.split(X, y, g))
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xte_s = scaler.transform(X[te])
    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr_s, y[tr])
    preds = model.predict(Xte_s)
    return model, scaler, {
        "mae": float(mean_absolute_error(y[te], preds)),
        "r2": float(r2_score(y[te], preds)),
        "n_train": len(tr),
        "n_test": len(te),
    }


def plot_tsne(feats, df, idxs, out_path):
    df_sub = df.loc[idxs].copy()
    phases = [str(p) for p in df_sub["days_until_molt"].astype(float).apply(lambda d: "imminent" if d <= 3 else "near" if d <= 7 else "mid" if d <= 14 else "far").tolist()]
    tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1), random_state=42, init="pca")
    emb = tsne.fit_transform(feats)
    cmap = {"imminent": "#d62728", "near": "#ff7f0e", "mid": "#1f77b4", "far": "#2ca02c"}
    plt.figure(figsize=(10, 8))
    for phase in sorted(set(phases)):
        mask = [p == phase for p in phases]
        plt.scatter(emb[mask, 0], emb[mask, 1], s=28, alpha=0.8, label=phase, c=cmap.get(phase, "#7f7f7f"))
    plt.legend(title="Molt phase")
    plt.title("SAM3 ventral (CLIP-filtered) t-SNE")
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved t-SNE to %s", out_path)


def make_gallery(df, idxs, out_path):
    sample = df.loc[idxs].sample(n=min(50, len(idxs)), random_state=42)
    cols = 10
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        ax.imshow(Image.open(row["image_path"]).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{row['crab_id']} | {row['days_until_molt']:.1f}d", fontsize=8)
    for ax in axes.flatten()[len(sample) :]:
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved gallery to %s", out_path)


def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    reports_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    feats, meta, df = load_data(root)
    idxs_all = meta["indices"]

    labels, scores_meta = score_ventral(feats, device="cpu", margin=0.05)
    ventral_mask = labels == 1
    if ventral_mask.sum() < 20:
        logger.warning("Only %d samples classified as ventral; results may be unstable.", int(ventral_mask.sum()))

    feats_v = feats[ventral_mask]
    idxs_v = [idx for idx, keep in zip(idxs_all, ventral_mask) if keep]

    # Persist metadata with filter decisions for inspection.
    meta_out = {
        "num_samples": int(feats_v.shape[0]),
        "feature_dim": int(feats_v.shape[1]),
        "indices": idxs_v,
        "orientation_labels": labels.tolist(),
        "scores": scores_meta,
    }
    (models_dir / "sam3_ventral_filtered_metadata.json").write_text(json.dumps(meta_out, indent=2))

    labels_days = df.loc[idxs_v, "days_until_molt"].astype(float).values
    groups = df.loc[idxs_v, "crab_id"].astype(str).values

    model_single, scaler_single, single_metrics = train_single(feats_v, labels_days, groups)
    joblib.dump(model_single, models_dir / "sam3_ventral_filtered_regressor.joblib")
    joblib.dump(scaler_single, models_dir / "sam3_ventral_filtered_scaler.joblib")
    logger.info(
        "SAM3 ventral (filtered) single-image MAE %.3f R2 %.3f",
        single_metrics["mae"],
        single_metrics["r2"],
    )

    window = 5
    X_seq, y_seq, g_seq = build_sequences(df, feats_v, idxs_v, window)
    if len(X_seq) > 1 and len(set(g_seq)) > 1:
        model_temp, scaler_temp, temp_metrics = train_temporal(X_seq, y_seq, g_seq)
        joblib.dump(model_temp, models_dir / f"sam3_ventral_filtered_temporal_w{window}.joblib")
        joblib.dump(scaler_temp, models_dir / f"sam3_ventral_filtered_temporal_scaler_w{window}.joblib")
        logger.info(
            "SAM3 ventral (filtered) temporal w=%d MAE %.3f R2 %.3f",
            window,
            temp_metrics["mae"],
            temp_metrics["r2"],
        )
    else:
        temp_metrics = None
        logger.info("Not enough filtered sequences for temporal model (window=%d); skipping.", window)

    metrics = {
        "orientation_counts": {
            "ventral": int((labels == 1).sum()),
            "dorsal": int((labels == 0).sum()),
            "uncertain": int((labels == -1).sum()),
        },
        "single_image": single_metrics,
        "temporal_w5": temp_metrics,
        "meta": meta_out,
    }
    (reports_dir / "sam3_ventral_filtered_metrics.json").write_text(json.dumps(metrics, indent=2))

    if len(feats_v) >= 2:
        plot_tsne(feats_v, df, idxs_v, plots_dir / "sam3_ventral_filtered_tsne.png")
    if len(idxs_v) > 0:
        make_gallery(df, idxs_v, plots_dir / "sam3_ventral_filtered_gallery.png")


if __name__ == "__main__":
    main()
