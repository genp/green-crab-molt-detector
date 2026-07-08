"""
Extractor protocol and shared parsing helpers.

An extractor turns one year's raw/processed sources into a list of normalized
``ImageRecord`` objects. Each concrete extractor lives in its own module and is
registered in ``extractors/__init__.py`` under a string key referenced by
``config.YearConfig.extractor``.

Design rules:
- Extractors must NOT import torch or any ML dependency; the extract stage is
  meant to be cheap and side-effect free.
- Every image that physically exists gets exactly one record, even if it has no
  labels — a blank-but-present row is better than a silently dropped image.
- All values are stored as clean strings via ``records.normalize_str``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from ..config import IMAGE_EXTENSIONS, REPO_ROOT, YearConfig
from ..records import ImageRecord, relpath


class Extractor(Protocol):
    """Callable that produces records for one year."""

    def __call__(self, cfg: YearConfig) -> List[ImageRecord]:
        ...


# --------------------------------------------------------------------------- #
# Filesystem helpers                                                            #
# --------------------------------------------------------------------------- #

def iter_images(root: Path) -> Iterable[Path]:
    """Yield every image under ``root`` (recursive), sorted for determinism."""
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    # De-dup (case-insensitive globs can double-count on case-insensitive FS).
    seen = set()
    unique = []
    for f in sorted(files):
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def make_relpath(abs_path: Path) -> str:
    return relpath(abs_path, REPO_ROOT)


# --------------------------------------------------------------------------- #
# Text/label parsing helpers (shared across year extractors)                    #
# --------------------------------------------------------------------------- #

RE_CRAB_ID = re.compile(r"\b([MF])\s?-?\s?(\d{1,3})\b", re.IGNORECASE)
RE_MOLT_TAG = re.compile(r"molted[\s_-]*([0-9]{1,2})[.:/-]([0-9]{1,2})", re.IGNORECASE)
RE_DATE_INLINE = re.compile(r"\b([0-9]{1,2})[.:/-]([0-9]{1,2})(?:[.:/-]([0-9]{2,4}))?\b")


def parse_sex_from_text(text: str) -> str:
    """Return 'male'/'female'/'' from a folder or crate name."""
    low = text.lower()
    if "female" in low or "(f)" in low or "egg" in low:
        return "female"
    if "male" in low or "(m)" in low:
        return "male"
    m = RE_CRAB_ID.search(text)
    if m:
        return "male" if m.group(1).upper() == "M" else "female"
    return ""


def parse_crab_id(text: str) -> str:
    m = RE_CRAB_ID.search(text)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return ""


def parse_date(text: str, default_year: Optional[int]) -> Optional[datetime]:
    """Parse the first M/D(/Y) date found in ``text``."""
    m = RE_DATE_INLINE.search(text)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if m.group(3):
        year = int(m.group(3))
        year += 2000 if year < 100 else 0
    elif default_year is not None:
        year = default_year
    else:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def parse_outcome(text: str) -> str:
    low = text.lower()
    if "died" in low or "dead" in low:
        return "died"
    if "thrown back" in low or "thrownback" in low:
        return "thrown_back"
    if "molted" in low or "molt" in low:
        return "molted"
    return ""


def fmt_date(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else ""


def days_between(capture: Optional[datetime], molt: Optional[datetime]) -> str:
    if capture is None or molt is None:
        return ""
    return str((molt - capture).days)
