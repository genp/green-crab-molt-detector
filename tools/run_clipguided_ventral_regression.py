#!/usr/bin/env python3
"""
YOLO detect -> multiple ventral candidate crops -> CLIP prompt scoring to select the best crop.
Extract OpenCLIP ViT-H/14 features, train regressors, t-SNE.

Outputs:
- models/clipguided_features.npy
- models/clipguided_metadata.json
- models/clipguided_regressor.joblib (+ scaler)
- models/clipguided_temporal_w5.joblib (+ scaler)
- reports/clipguided_metrics.json
- plots/clipguided_tsne.png
- plots/clipguided_crop_gallery.png
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
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


POS_PROMPT = "close-up photo of crab underside ventral plates, abdominal flap, high detail"
NEG_PROMPTS = [
    "human hand or fingers holding crab",
    "bucket or container background",
    "dorsal shell top view of crab",
]


def phase_from_days(days: float) -> str:
    if days <= 3:
        return "imminent"
    if days <= 7:
        return "near"
    if days <= 14:
        return "mid"
    return "far"


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["days_until_molt"].notna()].copy()
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["crab_id"] = df["crab_id"].astype(str)
    df["phase"] = df["days_until_molt"].astype(float).apply(phase_from_days)
    return df


def detect_crop(img: Image.Image, detector: YOLO, pad: float = 0.02) -> Image.Image:
    w, h = img.size
    res = detector(img, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return img
    areas = res.boxes.xyxy[:, 2:] - res.boxes.xyxy[:, :2]
    areas = areas[:, 0] * areas[:, 1]
    idx = int(torch.argmax(areas))
    x1, y1, x2, y2 = res.boxes.xyxy[idx].cpu().numpy()
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw
    x2 += pad * bw
    y1 -= pad * bh
    y2 += pad * bh
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def ventral_candidates(img: Image.Image) -> List[Image.Image]:
    """Generate a few ventral-focused candidates (lean/fast)."""
    w, h = img.size
    crops = []
    for frac in [0.5, 0.65]:
        top = int((1 - frac) * h)
        top = max(0, min(top, h - 2))
        crops.append(img.crop((0, top, w, h)))
    # center-bottom crop
    crops.append(img.crop((int(0.1 * w), int(0.55 * h), int(0.9 * w), h)))
    return crops


def clip_rank_crops(crops: List[Image.Image], model, tokenizer, preprocess, device: str, margin_thresh: float = 0.15) -> Image.Image:
    texts = [POS_PROMPT] + NEG_PROMPTS
    text_tok = tokenizer(texts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tok)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    pos_vec = text_emb[0:1]
    neg_vecs = text_emb[1:]

    best_crop = crops[0]
    best_score = -1e9
    for c in crops:
        with torch.no_grad():
            img_t = preprocess(c).unsqueeze(0).to(device)
            img_emb = model.encode_image(img_t)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            pos_score = (img_emb @ pos_vec.T).item()
            neg_score = (img_emb @ neg_vecs.T).mean().item()
            score = pos_score - neg_score
        if score > best_score:
            best_score = score
            best_crop = c
    # if margin is too low, fall back to bottom crop (last candidate)
    if best_score < margin_thresh:
        best_crop = crops[-1]
    return best_crop


def extract_features(df: pd.DataFrame, device: str = "cpu") -> Tuple[np.ndarray, List[int]]:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()
    detector = YOLO("yolov8n.pt")

    feats, idxs = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CLIP-guided crops"):
        img = Image.open(row["image_path"]).convert("RGB")
        det = detect_crop(img, detector, pad=0.0)
        candidates = ventral_candidates(det)
        best = clip_rank_crops(candidates, model, tokenizer, preprocess, device)
        with torch.no_grad():
            t = preprocess(best).unsqueeze(0).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().squeeze().astype(np.float32))
        idxs.append(idx)
    return np.stack(feats), idxs


def train_single(X, y, g, model_path, scaler_path):
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
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return {
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
    return np.stack(seqs), np.array(targs), np.array(groups)


def train_temporal(X, y, g, model_path, scaler_path):
    split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr, te = next(split.split(X, y, g))
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xte_s = scaler.transform(X[te])
    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr_s, y[tr])
    preds = model.predict(Xte_s)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return {
        "mae": float(mean_absolute_error(y[te], preds)),
        "r2": float(r2_score(y[te], preds)),
        "n_train": len(tr),
        "n_test": len(te),
    }


def plot_tsne(feats, df, idxs, out_path):
    df_sub = df.loc[idxs].copy()
    phases = [str(p) for p in df_sub["phase"].tolist()]
    tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1), random_state=42, init="pca")
    emb = tsne.fit_transform(feats)
    cmap = {"imminent": "#d62728", "near": "#ff7f0e", "mid": "#1f77b4", "far": "#2ca02c"}
    plt.figure(figsize=(10, 8))
    for phase in sorted(set(phases)):
        mask = [p == phase for p in phases]
        plt.scatter(emb[mask, 0], emb[mask, 1], s=28, alpha=0.8, label=phase, c=cmap.get(phase, "#7f7f7f"))
    plt.legend(title="Molt phase")
    plt.title("CLIP-guided ventral t-SNE")
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
    random.seed(42)
    np.random.seed(42)
    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "data" / "processed" / "manifest_with_2016_labels.csv"
    models_dir = root / "models"
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    df = load_manifest(manifest_path)
    logger.info("Labeled rows: %d, crabs: %d", len(df), df["crab_id"].nunique())

    feat_file = models_dir / "clipguided_features.npy"
    meta_file = models_dir / "clipguided_metadata.json"
    if feat_file.exists() and meta_file.exists():
        logger.info("Loading cached CLIP-guided features")
        feats = np.load(feat_file)
        meta = json.loads(meta_file.read_text())
        idxs = meta["indices"]
    else:
        feats, idxs = extract_features(df, device="cpu")
        np.save(feat_file, feats)
        meta = {"num_samples": int(feats.shape[0]), "feature_dim": int(feats.shape[1]), "indices": idxs}
        meta_file.write_text(json.dumps(meta, indent=2))

    labels = df.loc[idxs, "days_until_molt"].astype(float).values
    groups = df.loc[idxs, "crab_id"].astype(str).values

    single = train_single(
        feats, labels, groups, models_dir / "clipguided_regressor.joblib", models_dir / "clipguided_scaler.joblib"
    )
    logger.info("Single-image (clip-guided) MAE %.3f R2 %.3f", single["mae"], single["r2"])

    window = 5
    X_seq, y_seq, g_seq = build_sequences(df, feats, idxs, window)
    temporal = train_temporal(
        X_seq,
        y_seq,
        g_seq,
        models_dir / f"clipguided_temporal_w{window}.joblib",
        models_dir / f"clipguided_temporal_scaler_w{window}.joblib",
    )
    logger.info("Temporal (clip-guided, w=%d) MAE %.3f R2 %.3f", window, temporal["mae"], temporal["r2"])

    metrics = {"single_image": single, "temporal_w5": temporal, "meta": meta}
    (reports_dir / "clipguided_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_tsne(feats, df, idxs, plots_dir / "clipguided_tsne.png")
    make_gallery(df, idxs, plots_dir / "clipguided_crop_gallery.png")


if __name__ == "__main__":
    main()
