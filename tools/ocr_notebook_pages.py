#!/usr/bin/env python3
"""
Regenerate the 2026 paper-notebook transcription with a Claude vision model.

This is the reproducible, script-driven version of the notebook OCR: it sends the
photographed data-sheet pages (IMG_3637/3638/3639.HEIC, captured 2026-06-05) to
Claude via the Anthropic API and parses the handwriting into the same per-cell
CSV that `label_sheet/extractors/year_2026.py` consumes:

    data/curated/notebook_2026-06-05_transcription.csv

Handwriting OCR is not classical/deterministic — it's a vision model reading the
pages — so the output remains a *reviewable* artifact (re-runs can differ), but
it is now regenerable by command rather than by hand. Structured Outputs
(`output_config.format`) constrain the response to a fixed JSON schema so the
result parses cleanly every time.

Auth: uses the standard Anthropic credential chain (Anthropic() with no args).
If `ANTHROPIC_API_KEY` is unset, run `ant auth login` first (the zero-arg client
picks up the profile). Requires `pip install anthropic` and, for HEIC input,
`pillow-heif` (already in venv).

Usage:
    venv/bin/python tools/ocr_notebook_pages.py                       # defaults
    venv/bin/python tools/ocr_notebook_pages.py --pages IMG_3637.HEIC IMG_3638.HEIC IMG_3639.HEIC
    venv/bin/python tools/ocr_notebook_pages.py --model claude-opus-4-8 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
ALBUM_DIR = REPO_ROOT / "data" / "raw" / "Green Crab AI 2026"
OUT_CSV = REPO_ROOT / "data" / "curated" / "notebook_2026-06-05_transcription.csv"

DEFAULT_PAGES = ["IMG_3637.HEIC", "IMG_3638.HEIC", "IMG_3639.HEIC"]
DEFAULT_SESSION_DATE = "2026-06-05"

# JSON schema the model must fill (Structured Outputs). Note the constraints the
# API supports: object types need additionalProperties:false + required.
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["session_date", "observations"],
    "properties": {
        "session_date": {"type": "string", "description": "ISO date on the page header, e.g. 2026-06-05"},
        "salinity_ppt": {"type": "string"},
        "temp_c": {"type": "string"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["condo", "cell", "note", "empty"],
                "properties": {
                    "condo": {"type": "string", "description": "e.g. JEL-A, JEL-B, JEL-C"},
                    "cell": {"type": "string", "description": "grid cell A1..F6"},
                    "note": {"type": "string", "description": "verbatim molt-cue text; empty string if the cell is empty"},
                    "empty": {"type": "boolean", "description": "true if the cell is marked empty / has no crab"},
                },
            },
        },
    },
}

PROMPT = """These images are consecutive pages of a field notebook recording green crab
molt observations for a single monitoring session. The crabs live in "condos"
named JEL-A, JEL-B, and JEL-C; each condo is a 6x6 grid of cells labelled A1..F6
(columns A-F, rows 1-6).

Read the handwriting carefully and transcribe EVERY per-cell entry across all the
pages, in order. For each cell record:
- condo (JEL-A / JEL-B / JEL-C) — track which condo section you are in; a page may
  start partway through a condo and a "JEL X - starting HH:MM" line marks a new one,
- cell (e.g. A1, B3, F6),
- note: the exact molt-cue text written for that cell (seam / halo / dusky / crack /
  "no seam" / etc.). If the cell is written as "empty", set note to "" and empty=true.
- empty: true only when the entry says the cell is empty / has no crab.

Also capture the header session_date (e.g. "6/5/26" -> 2026-06-05) and any
salinity / temperature written at the top. Transcribe verbatim; do not summarize
or infer a phase. If a word is illegible, transcribe your best guess and append
"(?)"."""


def _load_image_b64(path: Path) -> tuple[str, str]:
    """Return (media_type, base64_data). Converts HEIC to JPEG for the API."""
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg", base64.standard_b64encode(path.read_bytes()).decode()
    if path.suffix.lower() == ".png":
        return "image/png", base64.standard_b64encode(path.read_bytes()).decode()
    # HEIC (and anything else) -> re-encode to JPEG via Pillow (+pillow-heif).
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        # Cap the long edge so the request stays well under size limits.
        rgb.thumbnail((2000, 2000))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode()


def _write_csv(session_date: str, observations: list, source_label: str, model: str) -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_date", "condo", "cell", "notebook_note", "empty",
                    "source_image", "transcription_method"])
        n = 0
        for obs in observations:
            note = (obs.get("note") or "").strip()
            empty = bool(obs.get("empty")) or note == ""
            w.writerow([
                obs.get("session_date") or session_date,
                obs.get("condo", ""),
                (obs.get("cell", "") or "").upper(),
                "" if empty else note,
                "1" if empty else "0",
                source_label,
                f"vision_ocr:{model} (needs expert review)",
            ])
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES,
                        help="Page image filenames under the album dir (HEIC/JPEG).")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--session-date", default=DEFAULT_SESSION_DATE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Encode images and print the request shape, but do not call the API.")
    args = parser.parse_args()

    page_paths = [ALBUM_DIR / p for p in args.pages]
    missing = [p for p in page_paths if not p.exists()]
    if missing:
        print(f"[ocr] missing page images: {missing}", file=sys.stderr)
        return 1

    content = []
    for p in page_paths:
        media_type, data = _load_image_b64(p)
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data}})
    content.append({"type": "text", "text": PROMPT})

    source_label = "/".join(p.name for p in page_paths)
    if args.dry_run:
        print(f"[ocr] {len(page_paths)} images encoded; model={args.model}; "
              f"prompt {len(PROMPT)} chars; would call messages.create with "
              f"output_config.format=json_schema. (dry run, no API call)")
        return 0

    try:
        import anthropic
    except ImportError:
        print("[ocr] pip install anthropic (not in this env). Use the venv or add the dep.",
              file=sys.stderr)
        return 2

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an `ant auth login` profile
    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        print(f"[ocr] request refused: {response.stop_details}", file=sys.stderr)
        return 3

    import json
    text = next((b.text for b in response.content if b.type == "text"), "")
    parsed = json.loads(text)
    n = _write_csv(parsed.get("session_date") or args.session_date,
                   parsed.get("observations", []), source_label, args.model)
    print(f"[ocr] wrote {n} cell observations -> {OUT_CSV}")
    print(f"[ocr] session_date={parsed.get('session_date')} "
          f"salinity={parsed.get('salinity_ppt')} temp={parsed.get('temp_c')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
