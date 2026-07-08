#!/usr/bin/env python3
"""
CLI for the centralized green crab label sheet.

Thin wrapper over ``label_sheet.pipeline``. See docs/LABEL_SHEET_PIPELINE.md.

Examples:
    # everything (extract labels, run cached model inference, build workbook)
    venv/bin/python tools/build_label_sheet.py all

    # stages
    venv/bin/python tools/build_label_sheet.py extract
    venv/bin/python tools/build_label_sheet.py predict --limit 50
    venv/bin/python tools/build_label_sheet.py assemble

    # skip ML entirely (thumbnails + labels only)
    venv/bin/python tools/build_label_sheet.py all --no-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so ``import label_sheet`` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_sheet import pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["extract", "predict", "assemble", "all"])
    parser.add_argument("--no-models", action="store_true",
                        help="Skip model inference; build labels+thumbnails only.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap images per model in the predict stage (for smoke tests).")
    args = parser.parse_args()

    if args.stage == "extract":
        pipeline.stage_extract()
    elif args.stage == "predict":
        pipeline.stage_predict(limit=args.limit)
    elif args.stage == "assemble":
        pipeline.stage_assemble(use_models=not args.no_models)
    elif args.stage == "all":
        pipeline.run_all(use_models=not args.no_models, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
