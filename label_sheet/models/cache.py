"""
Incremental prediction cache.

Predictions are stored in a long-format CSV keyed by
``(image_relpath, model_id)`` so the predict stage is resumable and idempotent:
re-running only computes pairs not already cached. Adding a model or a year does
not invalidate existing predictions.

Columns: image_relpath, model_id, <all Prediction fields...>
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd

from .registry import Prediction

PREDICTION_FIELDS = [
    "days_to_molt", "days_to_molt_std", "phase", "confidence", "sex",
    "view", "molt_indicators", "estimate_input", "in_training_set",
]
CACHE_COLUMNS = ["image_relpath", "model_id", *PREDICTION_FIELDS]


class PredictionCache:
    """Load/append/query the long-format prediction cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.df = pd.read_csv(path, dtype=str).fillna("")
            # Tolerate older caches missing newly added columns.
            for col in CACHE_COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = ""
        else:
            self.df = pd.DataFrame(columns=CACHE_COLUMNS)
        self._done: Set[Tuple[str, str]] = set(
            zip(self.df["image_relpath"], self.df["model_id"])
        )
        self._pending_rows = []

    def has(self, image_relpath: str, model_id: str) -> bool:
        return (image_relpath, model_id) in self._done

    def add(self, image_relpath: str, model_id: str, pred: Prediction) -> None:
        row = {"image_relpath": image_relpath, "model_id": model_id}
        for f in PREDICTION_FIELDS:
            row[f] = getattr(pred, f)
        self._pending_rows.append(row)
        self._done.add((image_relpath, model_id))

    def flush(self) -> None:
        """Persist pending rows to disk (append-safe full rewrite)."""
        if not self._pending_rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(self._pending_rows, columns=CACHE_COLUMNS)
        self.df = pd.concat([self.df, new], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        self._pending_rows = []

    def wide_lookup(self) -> Dict[str, Dict[str, str]]:
        """Return ``{image_relpath: {"m1_days_to_molt": ..., "m2_...": ...}}``.

        Prefixes each field with its model_id so the assemble stage can drop the
        values straight into schema columns.
        """
        out: Dict[str, Dict[str, str]] = {}
        for _, r in self.df.iterrows():
            rel = r["image_relpath"]
            mid = r["model_id"]
            bucket = out.setdefault(rel, {})
            for f in PREDICTION_FIELDS:
                bucket[f"{mid}_{f}"] = r.get(f, "")
        return out
