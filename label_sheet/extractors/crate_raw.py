"""
Crate 1-3 extractor — reads the RAW ``.docx`` files directly.

This deliberately does NOT depend on ``data/processed/crate_docs.csv`` or the
pre-extracted ``data/processed/crate_images/``. It parses each raw
``Crate N - Completed/Crab *.docx`` in place using only the stdlib (``zipfile`` +
``xml.etree``), so the label sheet is reproducible from raw data alone and needs
no ``python-docx`` dependency.

Per crab doc we recover:
- the metadata table row: crab number, carapace width, color, degree-of-molt,
- the body date paragraphs (capture dates, e.g. "6/20; 6/24; ..."),
- the embedded photos (``word/media/*``), written once to our own pipeline work
  dir ``data/processed/label_sheet/crate_images/`` (an intermediate produced from
  raw by this code, not a pre-existing artifact).

``degree_of_molt`` (imminent/late/mid/new) is mapped to an approximate
``days_until_molt`` with the same heuristic used elsewhere in the repo
(``tools/merge_crate_data.py``); this mapping is ours and is documented, not a
value read from a processed file.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from PIL import Image

from ..config import RAW_DIR, WORK_DIR, YearConfig
from ..records import ImageRecord, normalize_str
from .base import make_relpath

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

CRATE_DIRS = ["Crate 1 - Completed", "Crate 2 - Completed", "Crate 3 - Completed"]
CRATE_IMAGE_OUT = WORK_DIR / "crate_images"

DEGREE_TO_DAYS = {"imminent": "0.5", "late": "4", "mid": "9", "new": "16"}
DEGREE_TO_PHASE = {
    "imminent": "peeler_imminent", "late": "pre_molt",
    "mid": "pre_molt", "new": "molted",
}


def _first_table_row(root: ET.Element) -> Tuple[str, str, str, str]:
    """Return (crab_number, carapace, color, degree) from the first data row."""
    for tbl in root.iter(f"{W}tbl"):
        rows = list(tbl.iter(f"{W}tr"))
        if len(rows) < 2:
            return "", "", "", ""
        cells = []
        for c in rows[1].iter(f"{W}tc"):
            cells.append("".join(t.text or "" for t in c.iter(f"{W}t")).strip())
        cells += [""] * (4 - len(cells))
        return cells[0], cells[1], cells[2], cells[3]
    return "", "", "", ""


def _body_paragraphs(root: ET.Element) -> List[str]:
    """Body <w:p> text that is NOT inside a table (the capture-date lines)."""
    body = root.find(f"{W}body")
    if body is None:
        return []
    out = []
    for child in body:
        if child.tag == f"{W}p":
            txt = "".join(t.text or "" for t in child.iter(f"{W}t")).strip()
            if txt:
                out.append(txt)
    return out


def _extract_images(docx_path: Path, out_dir: Path) -> List[Path]:
    """Extract embedded media to ``out_dir`` as JPEGs; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    with zipfile.ZipFile(docx_path) as zf:
        for name in zf.namelist():
            if not name.startswith("word/media/"):
                continue
            data = zf.read(name)
            target = out_dir / f"{Path(name).stem}.jpg"
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.convert("RGB").save(target, format="JPEG", quality=90)
            except Exception:
                target = out_dir / Path(name).name
                target.write_bytes(data)
            saved.append(target)
    return sorted(saved)


def _looks_like_crab_doc(name: str) -> bool:
    low = name.lower()
    return low.startswith("crab") and not low.startswith(("observations", "salinity"))


def extract(cfg: YearConfig) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for crate_name in CRATE_DIRS:
        crate_dir = RAW_DIR / crate_name
        if not crate_dir.exists():
            continue
        for docx_path in sorted(crate_dir.glob("*.docx")):
            if not _looks_like_crab_doc(docx_path.name):
                continue
            try:
                with zipfile.ZipFile(docx_path) as zf:
                    root = ET.fromstring(zf.read("word/document.xml"))
            except Exception:
                continue

            crab_num, carapace, color, degree = _first_table_row(root)
            # Date-like body paragraphs only (drop stray header text).
            dates = [p for p in _body_paragraphs(root)
                     if any(ch.isdigit() for ch in p) and "/" in p]

            out_dir = CRATE_IMAGE_OUT / crate_name / docx_path.stem
            images = _extract_images(docx_path, out_dir)
            if not images:
                continue

            crab_id = f"{crate_name}::Crab{crab_num}" if crab_num else f"{crate_name}::{docx_path.stem}"
            degree_l = degree.strip().lower()
            for idx, img_path in enumerate(images):
                capture = dates[idx] if idx < len(dates) else ""
                note = "" if capture else (f"capture dates: {'; '.join(dates)}" if dates else "")
                records.append(ImageRecord(
                    # Crate study is the 2019 experiment (see salinity/observations
                    # docx dates); stamp the calendar year even though the sheet
                    # tab is named separately.
                    year="2019",
                    dataset=crate_name,
                    image_relpath=make_relpath(img_path),
                    abs_path=str(img_path),
                    crab_id=crab_id,
                    observation_id=crab_id,
                    session_id=f"{crate_name}::{crab_num}",
                    original_filename=f"{docx_path.stem}/{img_path.name}",
                    capture_date=capture,
                    days_until_molt=DEGREE_TO_DAYS.get(degree_l, ""),
                    known_molt_phase=DEGREE_TO_PHASE.get(degree_l, ""),
                    color_state=color,
                    carapace_width_mm=carapace,
                    degree_of_molt=degree,
                    label_source=f"raw docx: {crate_name}/{docx_path.name}",
                    label_confidence="low" if degree_l else "unknown",
                    source_notes=note,
                    is_in_situ="false",
                ))
    return records
