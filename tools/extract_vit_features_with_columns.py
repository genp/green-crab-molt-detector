"""
Extract ViT features per image and embed them as feature_* columns alongside metadata.

Input: data/processed/crab_dataset.csv (2016-only loader output)
Output: data/processed/crab_dataset_2016_vit.csv with feature_0..N columns
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append("src")
from feature_extractor import GeneralCrustaceanFeatureExtractor


def main():
    dataset_path = Path("data/processed/crab_dataset.csv")
    out_path = Path("data/processed/crab_dataset_2016_vit.csv")
    df = pd.read_csv(dataset_path)

    extractor = GeneralCrustaceanFeatureExtractor("vit_base", device="cpu")
    feats = []
    for p in tqdm(df["image_path"], desc="Extracting ViT features"):
        feats.append(extractor.extract_features(p))
    feats = np.stack(feats)

    # Add feature columns
    feat_cols = [f"feature_{i}" for i in range(feats.shape[1])]
    feat_df = pd.DataFrame(feats, columns=feat_cols)
    out_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {out_df.shape}")


if __name__ == "__main__":
    main()
