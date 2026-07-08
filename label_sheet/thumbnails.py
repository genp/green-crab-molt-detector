"""
Cached, EXIF-correct thumbnail generation.

Thumbnails are written once to ``config.THUMB_CACHE_DIR`` and reused across
workbook rebuilds. The cache key is a hash of the absolute source path plus the
requested size, so moving/renaming a source image naturally invalidates it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from .config import THUMB_CACHE_DIR

THUMB_SIZE = 96


def _cache_path(src: Path, size: int) -> Path:
    digest = hashlib.sha1(f"{src.resolve()}::{size}".encode("utf-8")).hexdigest()[:16]
    return THUMB_CACHE_DIR / f"{digest}.jpg"


def get_thumbnail(src_path: str | Path, size: int = THUMB_SIZE) -> Optional[Path]:
    """Return a path to a cached square thumbnail, or None if the source is
    missing or unreadable.

    The thumbnail is letterboxed onto a white ``size``x``size`` canvas so every
    row in the workbook has a uniform image box.
    """
    src = Path(src_path)
    if not src.exists():
        return None
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _cache_path(src, size)
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        with Image.open(src) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), "white")
            offset = ((size - image.width) // 2, (size - image.height) // 2)
            canvas.paste(image, offset)
            canvas.save(out, format="JPEG", quality=82)
        return out
    except Exception:
        return None
