"""
Auxiliary-label taggers: sex, view, and visible molt indicators.

These produce the aux labels the request asks the models to estimate alongside
days-to-molt. They are deliberately transparent and swappable:

- ``molt_indicators`` are REAL color/brightness descriptors computed from the
  image (translucency, ventral color, dark seam presence). Molt progression in
  green crabs shows up as ventral color shifting green->yellow->orange->red and
  as a developing suture/seam line, so color statistics are a legitimate,
  explainable proxy. These tags are meant to prompt the expert, not replace them.
- ``sex`` and ``view`` are best-effort heuristics with honest ``unknown``
  fallbacks. They are structured behind :class:`AuxTagger` so a trained
  classifier can replace the heuristic without touching callers.

Nothing here fabricates a precise label it cannot support; low-signal cases
return ``unknown`` and low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass
class AuxResult:
    sex: str = "unknown"
    view: str = "unknown"
    molt_indicators: str = ""
    view_confidence: str = "low"


def _load_rgb(image_path: str, max_side: int = 256) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as opened:
            im = ImageOps.exif_transpose(opened).convert("RGB")
            im.thumbnail((max_side, max_side))
            return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def _center_crop(arr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = arr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def _rgb_to_hsv_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    """Return mean (hue_deg, saturation, value) over the crop."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn + 1e-6
    hue = np.zeros_like(mx)
    mask = mx == r
    hue[mask] = (60 * ((g - b) / diff) % 360)[mask]
    mask = mx == g
    hue[mask] = (60 * ((b - r) / diff) + 120)[mask]
    mask = mx == b
    hue[mask] = (60 * ((r - g) / diff) + 240)[mask]
    sat = diff / (mx + 1e-6)
    return float(np.mean(hue)), float(np.mean(sat)), float(np.mean(mx))


class AuxTagger:
    """Heuristic aux-label tagger.

    Replace ``tag`` (or subclass) to drop in learned sex/view classifiers while
    keeping the same :class:`AuxResult` contract.
    """

    def tag(self, image_path: str) -> AuxResult:
        arr = _load_rgb(image_path)
        if arr is None:
            return AuxResult()
        crop = _center_crop(arr)
        hue, sat, val = _rgb_to_hsv_stats(crop)

        indicators = self._molt_indicators(hue, sat, val)
        view = self._guess_view(arr, hue, sat, val)
        # Sex is not reliably inferable from a whole image without the ventral
        # abdomen segmented; stay honest.
        return AuxResult(sex="unknown", view=view, molt_indicators="; ".join(indicators))

    @staticmethod
    def _molt_indicators(hue: float, sat: float, val: float) -> List[str]:
        tags: List[str] = []
        # Ventral/shell color along the intermolt->premolt progression.
        if sat < 0.12:
            tags.append("low-saturation/pale (possible translucency)")
        if val > 0.75 and sat < 0.25:
            tags.append("bright/buttery-opaque")
        if 15 <= hue <= 45 and sat >= 0.25:
            tags.append("orange/yellow ventral")
        elif hue < 15 or hue > 345:
            tags.append("reddish tone")
        elif 60 <= hue <= 160:
            tags.append("green tone")
        if val < 0.30:
            tags.append("dark regions (possible seam/suture)")
        if not tags:
            tags.append("no strong color cue")
        return tags

    @staticmethod
    def _guess_view(arr: np.ndarray, hue: float, sat: float, val: float) -> str:
        """Very weak view heuristic. Returns 'unknown' unless a cue is clear.

        Side views tend to be wide/short (high aspect); ventral tends to be
        brighter/paler (pale abdomen); dorsal tends to be greener/darker.
        """
        h, w = arr.shape[:2]
        aspect = w / max(h, 1)
        if aspect >= 1.7 or aspect <= 0.58:
            return "side"
        if val > 0.6 and sat < 0.35:
            return "ventral"
        if 60 <= hue <= 160 and val < 0.55:
            return "dorsal"
        return "unknown"
