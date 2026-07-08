"""
Days-to-molt estimator runner.

Wraps a feature extractor + a joblib regressor + the aux tagger into a
``ModelRunner``. Produces ``days_to_molt`` (and, for bootstrap models, a std),
maps days to a molt phase, and attaches aux labels.

Regressor artifacts in this repo come in two shapes:
- a bare sklearn estimator (e.g. ``openclip_regressor.joblib``), or
- a dict ``{"model", "scaler", ...}`` (e.g. ``molt_regressor_vit_random_forest``).
Both are handled. If a scaler is present it is applied before predict.

Bootstrap uncertainty: for a RandomForest we take the per-tree predictions and
report their std; this is a cheap, principled dispersion estimate that needs no
retraining. For non-forest models the std is left blank.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from ..config import MODELS_DIR, ModelConfig
from .auxlabels import AuxTagger
from .features import get_extractor
from .registry import ModelRunner, Prediction, register_runner


def _days_to_phase(days: float) -> str:
    """Map a continuous days-to-molt estimate to the schema phase vocabulary."""
    if days <= 0.5:
        return "molted"
    if days <= 3:
        return "peeler_imminent"
    if days <= 10:
        return "pre_molt"
    return "intermolt"


def _confidence_from_std(std: Optional[float]) -> str:
    if std is None:
        return "medium"
    if std <= 1.5:
        return "high"
    if std <= 4.0:
        return "medium"
    return "low"


class EstimatorRunner:
    """A days-to-molt estimator over one feature backend + one regressor."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.model_id = cfg.model_id
        self.cfg = cfg
        self._aux = AuxTagger()

        # Load the regressor artifact (dict or bare estimator).
        bundle = joblib.load(MODELS_DIR / cfg.weights)
        if isinstance(bundle, dict):
            self._model = bundle.get("model")
            self._scaler = bundle.get("scaler")
        else:
            self._model = bundle
            self._scaler = None
        # A bare estimator may have its scaler in a separate file (e.g. OpenCLIP).
        if self._scaler is None and cfg.scaler:
            self._scaler = joblib.load(MODELS_DIR / cfg.scaler)

        # Load the requested feature backend, with graceful OpenCLIP fallback.
        self._features = None
        self._feature_kind = cfg.feature_extractor
        try:
            self._features = get_extractor(cfg.feature_extractor)
        except Exception as exc:  # OpenCLIP weights/network unavailable, etc.
            if cfg.feature_extractor == "openclip":
                self._feature_kind = "vit"
                self._features = get_extractor("vit")
                self._fallback_note = f"openclip unavailable ({exc}); used vit"
            else:
                raise
        self._fallback_note = getattr(self, "_fallback_note", "")

        # Sanity: feature dim vs regressor input dim.
        n_in = getattr(self._model, "n_features_in_", None)
        self._dim_ok = (n_in is None) or (n_in == getattr(self._features, "dim", n_in))

    def predict_path(self, abs_path: str, in_training_set: bool) -> Prediction:
        aux = self._aux.tag(abs_path)
        pred = Prediction(
            sex=aux.sex,
            view=aux.view,
            molt_indicators=aux.molt_indicators,
            estimate_input="whole_image_fallback",
            in_training_set="true" if in_training_set else "false",
        )
        if not self._dim_ok:
            # Feature/regressor mismatch (e.g. openclip fallback into a 1024-d
            # regressor). Emit aux labels only, and say why.
            pred.confidence = "unknown"
            pred.molt_indicators = (
                (aux.molt_indicators + "; " if aux.molt_indicators else "")
                + f"[days skipped: {self._feature_kind} dim != regressor input]"
            )
            return pred

        try:
            feats = self._features.extract(abs_path).reshape(1, -1)
        except Exception:
            pred.confidence = "unknown"
            return pred

        X = self._scaler.transform(feats) if self._scaler is not None else feats

        mean, std = self._predict_with_std(X)
        pred.days_to_molt = f"{mean:.2f}"
        pred.phase = _days_to_phase(mean)
        if self.cfg.bootstrap and std is not None:
            pred.days_to_molt_std = f"{std:.2f}"
            pred.confidence = _confidence_from_std(std)
        else:
            pred.confidence = "medium"
        if self._fallback_note:
            pred.molt_indicators = (
                (pred.molt_indicators + "; " if pred.molt_indicators else "")
                + f"[{self._fallback_note}]"
            )
        return pred

    def _predict_with_std(self, X: np.ndarray):
        """Return (mean, std). std is None unless the model is a tree ensemble."""
        estimators = getattr(self._model, "estimators_", None)
        if self.cfg.bootstrap and estimators is not None:
            per_tree = np.array([est.predict(X)[0] for est in estimators], dtype=np.float64)
            return float(per_tree.mean()), float(per_tree.std())
        return float(self._model.predict(X)[0]), None


register_runner("estimator", EstimatorRunner)
