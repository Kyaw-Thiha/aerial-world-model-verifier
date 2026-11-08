"""Cheap per-frame image statistics for the error breakdowns (blur proxy, luma).

Motion blur has no direct label in TII beyond the per-flight `exposure` setting, so
this records a stated proxy: the variance of the Laplacian (higher = sharper) and
the mean luma, for the frames a GeoCalib run used. Local CPU, no GPU.

    pixi run -e gpu python scripts/frame_stats.py --tag div_zips_6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geocalib-root", default="data/raw/geocalib")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames-root", default="data/raw/tii_frames")
    args = ap.parse_args()
    import cv2

    src = Path(args.geocalib_root) / args.tag
    out = src / "frame_stats.json"
    stats = {}
    for jl in sorted(src.glob("*.jsonl")):
        flight = jl.stem
        fdir = Path(args.frames_root) / flight
        have = {p.name: p for p in fdir.glob("*.jpg")}
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            name = json.loads(line)["frame"]
            p = have.get(name)
            if p is None:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            stats[f"{flight}/{name}"] = {
                "lap_var": float(cv2.Laplacian(img, cv2.CV_64F).var()),
                "mean_luma": float(img.mean())}
    out.write_text(json.dumps(stats, indent=1))
    print(f"wrote {out} ({len(stats)} frames with local copies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
