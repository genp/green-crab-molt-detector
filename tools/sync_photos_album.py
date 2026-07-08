#!/usr/bin/env python3
"""
Copy an Apple Photos shared album into data/raw (idempotent).

The "Green Crab AI 2026" labels live in a shared Apple Photos album. This tool
exports that album's originals into ``data/raw/<album>`` so the rest of the
pipeline can treat it like any other on-disk dataset.

Two backends, tried in order:

1. **osxphotos** (preferred): ``pip install osxphotos``. Exports originals by
   album name, preserving filenames, skipping files that already exist.
2. **AppleScript/Photos** fallback: drives Photos.app to export the album to a
   temp dir, then syncs into the destination. Requires the Photos app and
   Automation permission for the terminal.

The copy is additive: existing files in the destination are never overwritten or
deleted, so re-running only fills in what's missing.

NOTE: This is intentionally NOT called by build_label_sheet.py — touching the
Photos library is an explicit, user-initiated action.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def have_osxphotos() -> bool:
    try:
        import osxphotos  # noqa: F401
        return True
    except Exception:
        return False


def export_with_osxphotos(album: str, dest: Path) -> int:
    """Export album originals via osxphotos. Returns number of files present."""
    import osxphotos

    dest.mkdir(parents=True, exist_ok=True)
    photosdb = osxphotos.PhotosDB()
    photos = [p for p in photosdb.photos() if album in (p.albums or [])]
    if not photos:
        print(f"[sync] osxphotos: no photos found in album {album!r}. "
              f"Known albums include: {sorted(set(a for p in photosdb.photos() for a in (p.albums or [])))[:20]}")
    exported = 0
    for photo in photos:
        target = dest / (photo.original_filename or photo.filename)
        if target.exists():
            continue
        try:
            photo.export(str(dest), photo.original_filename or photo.filename,
                         use_photos_export=True)
            exported += 1
        except Exception as exc:
            print(f"[sync]   failed {photo.original_filename}: {exc}")
    print(f"[sync] osxphotos exported {exported} new files into {dest}")
    return sum(1 for _ in dest.rglob("*") if _.is_file())


APPLESCRIPT_TEMPLATE = '''
tell application "Photos"
    set theAlbum to album "{album}"
    set theItems to media items of theAlbum
    export theItems to POSIX file "{tmpdir}" with using originals
end tell
'''


def export_with_applescript(album: str, dest: Path) -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="photos_export_"))
    script = APPLESCRIPT_TEMPLATE.format(album=album, tmpdir=str(tmpdir))
    print(f"[sync] driving Photos.app to export {album!r} -> {tmpdir}")
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[sync] AppleScript export failed:\n{proc.stderr}")
        return -1
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in tmpdir.rglob("*"):
        if src.is_file():
            target = dest / src.name
            if not target.exists():
                shutil.copy2(src, target)
                copied += 1
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[sync] AppleScript copied {copied} new files into {dest}")
    return sum(1 for _ in dest.rglob("*") if _.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--album", default="Green Crab AI 2026")
    parser.add_argument("--dest", default="data/raw/Green Crab AI 2026",
                        help="Destination folder (relative to repo root or absolute).")
    parser.add_argument("--backend", choices=["auto", "osxphotos", "applescript"],
                        default="auto")
    args = parser.parse_args()

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest

    existing = sum(1 for _ in dest.rglob("*") if _.is_file()) if dest.exists() else 0
    print(f"[sync] destination {dest} currently has {existing} files")

    backend = args.backend
    if backend == "auto":
        backend = "osxphotos" if have_osxphotos() else "applescript"

    if backend == "osxphotos":
        if not have_osxphotos():
            print("[sync] osxphotos not installed. Run: pip install osxphotos")
            return 1
        total = export_with_osxphotos(args.album, dest)
    else:
        total = export_with_applescript(args.album, dest)

    if total < 0:
        return 1
    print(f"[sync] done. destination now has {total} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
