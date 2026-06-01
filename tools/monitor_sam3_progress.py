#!/usr/bin/env python3
"""Write periodic progress snapshots for a SAM3 bbox bootstrap run."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


PROGRESS_RE = re.compile(r"SAM3 bbox proposals:\s+(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--total-images", type=int, default=5495)
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--notify-every-min", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def latest_progress_line(log_path: Path) -> Optional[tuple[int, int, int]]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    pct, done, total = matches[-1]
    return int(pct), int(done), int(total)


def notify(message: str) -> None:
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "SAM3 bootstrap"',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def snapshot(run_dir: Path, total_images: int) -> str:
    processed = count_csv_rows(run_dir / "processed_images.csv")
    proposals = count_csv_rows(run_dir / "proposals.csv")
    progress = latest_progress_line(run_dir / "run.log")
    if progress:
        pct, run_done, run_total = progress
        run_part = f"run_progress={run_done}/{run_total} ({pct}%)"
    else:
        run_part = "run_progress=unknown"
    overall_pct = (processed / total_images * 100) if total_images else 0
    return (
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"processed={processed}/{total_images} ({overall_pct:.1f}%) "
        f"proposals={proposals} {run_part}"
    )


def main() -> None:
    args = parse_args()
    progress_log = args.run_dir / "progress_monitor.log"
    last_notify = 0.0

    while True:
        line = snapshot(args.run_dir, args.total_images)
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

        if args.notify_every_min > 0:
            now = time.time()
            if now - last_notify >= args.notify_every_min * 60:
                notify(line)
                last_notify = now

        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
