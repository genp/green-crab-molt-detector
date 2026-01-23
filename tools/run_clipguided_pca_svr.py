#!/usr/bin/env python3
"""
Evaluate lightweight heads (PCA + SVR and ElasticNet) on the existing
CLIP-guided ventral features.

Inputs:
- models/clipguided_features.npy
- models/clipguided_metadata.json
- data/processed/manifest_with_2016_labels.csv

Outputs:
- reports/clipguided_pca_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def load_data(root: Path):
    feats = np.load(root / "models" / "clipguided_features.npy")
    meta = json.loads((root / "models" / "clipguided_metadata.json").read_text())
    df = pd.read_csv(root / "data" / "processed" / "manifest_with_2016_labels.csv")
    df = df[df["days_until_molt"].notna()].copy()
    labels = df.loc[meta["indices"], "days_until_molt"].astype(float).values
    groups = df.loc[meta["indices"], "crab_id"].astype(str).values
    return feats, labels, groups, meta


def evaluate_model(X, y, groups, make_model):
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr, te = next(splitter.split(X, y, groups))
    model, scaler = make_model()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    model.fit(Xtr, y[tr])
    preds = model.predict(Xte)
    return {
        "mae": float(mean_absolute_error(y[te], preds)),
        "r2": float(r2_score(y[te], preds)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    feats, labels, groups, meta = load_data(root)

    def make_svr():
        scaler = StandardScaler()
        pca = PCA(n_components=min(128, feats.shape[1]))
        model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")

        class Model:
            def fit(self, X, y):
                self.pca = pca.fit(X)
                self.model = model.fit(self.pca.transform(X), y)
                return self

            def predict(self, X):
                return self.model.predict(self.pca.transform(X))

        return Model(), scaler

    def make_enet():
        scaler = StandardScaler()
        pca = PCA(n_components=min(64, feats.shape[1]))
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)

        class Model:
            def fit(self, X, y):
                self.pca = pca.fit(X)
                self.model = model.fit(self.pca.transform(X), y)
                return self

            def predict(self, X):
                return self.model.predict(self.pca.transform(X))

        return Model(), scaler

    metrics = {
        "pca_svr": evaluate_model(feats, labels, groups, make_svr),
        "pca_elasticnet": evaluate_model(feats, labels, groups, make_enet),
        "meta": meta,
    }

    (reports_dir / "clipguided_pca_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
