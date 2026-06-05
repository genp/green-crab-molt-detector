#!/usr/bin/env python3
"""
Train a split-aware molt estimator from precomputed feature columns.

The script joins feature rows to data/processed/estimator_v2_manifest.csv by
image_path, uses the global manifest split, and reports train/val/test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/crab_dataset_2016_vit.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/estimator_v2_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/estimator_v2"))
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def main() -> int:
    args = parse_args()
    features_df = pd.read_csv(args.features)
    manifest_df = pd.read_csv(args.manifest)
    feature_cols = [col for col in features_df.columns if col.startswith("feature_")]
    if not feature_cols:
        raise SystemExit(f"No feature_* columns found in {args.features}")

    features_df["image_path"] = features_df["image_path"].astype(str)
    manifest_df["image_path"] = manifest_df["image_path"].astype(str)
    merged = manifest_df.merge(
        features_df[["image_path", *feature_cols]],
        on="image_path",
        how="inner",
        validate="many_to_one",
    )
    merged["days_to_molt"] = pd.to_numeric(merged["days_to_molt"], errors="coerce")
    merged = merged.dropna(subset=["days_to_molt"])
    if merged.empty:
        raise SystemExit("No estimator rows matched feature rows.")

    train_df = merged[merged["split"] == "train"].copy()
    eval_splits = {
        "train": train_df,
        "val": merged[merged["split"] == "val"].copy(),
        "test": merged[merged["split"] == "test"].copy(),
    }
    if train_df.empty:
        raise SystemExit("No train rows after joining manifest to features.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["days_to_molt"].to_numpy(dtype=np.float32)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {}
        for split_name, split_df in eval_splits.items():
            if split_df.empty:
                continue
            X = scaler.transform(split_df[feature_cols].to_numpy(dtype=np.float32))
            y = split_df["days_to_molt"].to_numpy(dtype=np.float32)
            pred = model.predict(X)
            results[name][split_name] = metrics(y, pred)
        joblib.dump({"model": model, "scaler": scaler, "feature_columns": feature_cols}, args.output_dir / f"{name}.joblib")

    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Matched rows: {len(merged)}")
    print(json.dumps(results, indent=2))
    print(f"Wrote models and results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
