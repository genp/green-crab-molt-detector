#!/usr/bin/env python3
"""
SAM3 text-prompt segmentation on single images -> ventral-focused crop -> OpenCLIP features ->
single-image RF and temporal GB regressors.

Requires the focal3.12 venv (transformers with SAM3). Run with:
~/.venv/focal3.12/bin/python tools/run_sam3_ventral_regression.py
"""

from __future__ import annotations

import json
import logging
import os
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
from transformers import Sam3VideoModel, Sam3VideoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = ["a close-up of a crab underside", "crab ventral plates"]


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


def mask_to_bbox(mask: torch.Tensor, orig_h: int, orig_w: int) -> Tuple[int, int, int, int]:
    # mask expected (H, W)
    if mask.dtype != torch.bool:
        mask = mask > 0
    if not mask.any():
        return 0, 0, orig_w, orig_h
    ys, xs = torch.nonzero(mask, as_tuple=True)
    y1 = ys.min().item() * orig_h / mask.shape[0]
    y2 = ys.max().item() * orig_h / mask.shape[0]
    x1 = xs.min().item() * orig_w / mask.shape[1]
    x2 = xs.max().item() * orig_w / mask.shape[1]
    return int(x1), int(y1), int(x2), int(y2)


def sam3_crop(img: Image.Image, model, processor, device: torch.device, pad: float = 0.06) -> Image.Image:
    np_frame = np.array(img)
    frames = np.expand_dims(np_frame, axis=0)  # (1, H, W, 3)
    sess = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device="cpu",
        dtype=torch.float32,
    )
    processor.add_text_prompt(sess, PROMPTS)
    with torch.no_grad():
        out = model(inference_session=sess, frame_idx=0)

    obj_ids = out["object_ids"] if "object_ids" in out else []
    # SAM3 may return a tensor or a Python list here.
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.cpu().tolist()
    elif not isinstance(obj_ids, list):
        obj_ids = list(obj_ids)
    best = None
    orig_w, orig_h = img.size
    for oid in obj_ids:
        mask = out["obj_id_to_mask"].get(int(oid))
        if mask is None:
            continue
        bbox = mask_to_bbox(mask.squeeze(0), orig_h, orig_w)
        score = float(out["obj_id_to_score"].get(int(oid), 0.0))
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        rank = score * area
        if best is None or rank > best[0]:
            best = (rank, bbox)

    if best is None:
        return img

    _, (x1, y1, x2, y2) = best
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(pad * bw))
    x2 = min(orig_w, x2 + int(pad * bw))
    y1 = max(0, y1 - int(pad * bh))
    y2 = min(orig_h, y2 + int(pad * bh))
    crop = img.crop((x1, y1, x2, y2))
    # ventral-focused: take bottom 55% of the crop
    cw, ch = crop.size
    top = int(ch * 0.45)
    return crop.crop((0, top, cw, ch))


def extract_features(df: pd.DataFrame, device: str = "cpu") -> Tuple[np.ndarray, List[int]]:
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model = clip_model.to(device)
    clip_model.eval()

    sam_device = torch.device(device)
    sam_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    sam_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(sam_device)
    sam_model.eval()

    feats, idxs = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="SAM3 ventral crops"):
        img = Image.open(row["image_path"]).convert("RGB")
        crop = sam3_crop(img, sam_model, sam_processor, sam_device)
        with torch.no_grad():
            t = preprocess(crop).unsqueeze(0).to(device)
            f = clip_model.encode_image(t)
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
    if not seqs:
        feat_dim = feats.shape[1]
        return (
            np.empty((0, feat_dim * window), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
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
    plt.title("SAM3 ventral t-SNE")
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
    max_images = int(os.environ.get("SAM3_MAX_IMAGES", "0"))
    if max_images > 0 and len(df) > max_images:
        # Sample a subset for quicker experimentation; keep stratified-ish by crab/time.
        df = (
            df.groupby("crab_id", group_keys=False)
            .apply(lambda g: g.sort_values("capture_date"))
            .sample(n=max_images, random_state=42)
            .sort_values(["crab_id", "capture_date"])
        )
    logger.info("Labeled rows: %d, crabs: %d", len(df), df["crab_id"].nunique())

    feat_file = models_dir / "sam3_ventral_features.npy"
    meta_file = models_dir / "sam3_ventral_metadata.json"
    force_recompute = os.environ.get("SAM3_FORCE_RECOMPUTE", "").lower() in {"1", "true", "yes"}
    if feat_file.exists() and meta_file.exists() and not force_recompute:
        logger.info("Loading cached SAM3 features")
        feats = np.load(feat_file)
        meta = json.loads(meta_file.read_text())
        idxs = meta["indices"]
    else:
        if force_recompute:
            logger.info("SAM3_FORCE_RECOMPUTE is set; recomputing SAM3 features.")
        feats, idxs = extract_features(df, device="cpu")
        np.save(feat_file, feats)
        meta = {"num_samples": int(feats.shape[0]), "feature_dim": int(feats.shape[1]), "indices": idxs}
        meta_file.write_text(json.dumps(meta, indent=2))

    labels = df.loc[idxs, "days_until_molt"].astype(float).values
    groups = df.loc[idxs, "crab_id"].astype(str).values

    single = train_single(
        feats, labels, groups, models_dir / "sam3_ventral_regressor.joblib", models_dir / "sam3_ventral_scaler.joblib"
    )
    logger.info("Single-image (SAM3 ventral) MAE %.3f R2 %.3f", single["mae"], single["r2"])

    window = 5
    X_seq, y_seq, g_seq = build_sequences(df, feats, idxs, window)
    if len(X_seq) > 0:
        temporal = train_temporal(
            X_seq,
            y_seq,
            g_seq,
            models_dir / f"sam3_ventral_temporal_w{window}.joblib",
            models_dir / f"sam3_ventral_temporal_scaler_w{window}.joblib",
        )
        logger.info("Temporal (SAM3 ventral, w=%d) MAE %.3f R2 %.3f", window, temporal["mae"], temporal["r2"])
    else:
        temporal = None
        logger.info("Not enough sequences for temporal model (window=%d); skipping.", window)

    metrics = {"single_image": single, "temporal_w5": temporal, "meta": meta}
    (reports_dir / "sam3_ventral_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_tsne(feats, df, idxs, plots_dir / "sam3_ventral_tsne.png")
    make_gallery(df, idxs, plots_dir / "sam3_ventral_gallery.png")


if __name__ == "__main__":
    main()
