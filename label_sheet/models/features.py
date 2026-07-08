"""
Image feature extractors for the estimator runners.

Two backends:
- ``vit``: torchvision ``vit_b_16`` penultimate features (768-d). This reuses
  ``src/feature_extractor.py:GeneralCrustaceanFeatureExtractor`` — the exact
  extractor the deployed app uses — so the primary (``m1``) model sees the same
  inputs it was trained/served on.
- ``openclip``: OpenCLIP ``ViT-H-14 / laion2b_s32b_b79k`` image embedding
  (1024-d), matching the newest OpenCLIP-based regressor artifacts.

Both expose ``extract(image_path) -> np.ndarray`` (1-D float32). Backends are
imported lazily so the extract stage never pulls in torch.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import REPO_ROOT

# Make src/ importable so we can reuse the app's ViT extractor verbatim.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class ViTFeatures:
    """torchvision vit_b_16 features (768-d), via the app's extractor."""

    dim = 768

    def __init__(self) -> None:
        from feature_extractor import GeneralCrustaceanFeatureExtractor  # type: ignore
        self._impl = GeneralCrustaceanFeatureExtractor("vit_base")

    def extract(self, image_path: str) -> np.ndarray:
        vec = self._impl.extract_features(image_path)
        return np.asarray(vec, dtype=np.float32).reshape(-1)


class OpenCLIPFeatures:
    """OpenCLIP ViT-H-14 image embedding (1024-d)."""

    dim = 1024
    MODEL_NAME = "ViT-H-14"
    PRETRAINED = "laion2b_s32b_b79k"

    def __init__(self) -> None:
        import open_clip  # type: ignore
        import torch

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED)
        model.eval().to(device)
        self._model = model
        self._preprocess = preprocess

    def extract(self, image_path: str) -> np.ndarray:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().reshape(-1).astype(np.float32)


@lru_cache(maxsize=None)
def get_extractor(kind: str):
    """Return a cached feature extractor instance for ``kind``.

    Cached so multiple runners sharing a backend don't reload weights, and so a
    failed OpenCLIP load can be caught once by the caller.
    """
    if kind == "vit":
        return ViTFeatures()
    if kind == "openclip":
        return OpenCLIPFeatures()
    raise KeyError(f"Unknown feature extractor {kind!r}")
