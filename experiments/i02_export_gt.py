"""Export the set's ground truth at the video rate, and a summary of its composition.

Two outputs, both deterministic from ``configs/sim_splits_frozen.json`` and needing no render
and no GPU:

* ``results/i02/gt_video_rate.csv`` -- one row per video frame per clip: the required and the
  depicted lean (2-D each), their tilts in degrees and a defect-active flag. This is what a
  clip-level or segment-level reader can line up against the video without unpacking the
  250 Hz truth on R2.
* ``results/i02/composition.json`` -- counts, tilt distributions and the defect magnitudes per
  split and family, the numbers the report quotes.

    python experiments/i02_export_gt.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.sim.dataset import build_clip

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "configs/sim_splits_frozen.json"
OUT_CSV = ROOT / "results/i02/gt_video_rate.csv"
OUT_JSON = ROOT / "results/i02/composition.json"

FIELDS = ["clip_id", "frame", "t", "lean_req_x", "lean_req_y", "lean_dep_x", "lean_dep_y",
          "tilt_req_deg", "tilt_dep_deg", "defect_active"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(TABLE))
    args = ap.parse_args()
    table = json.loads(Path(args.table).read_text())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    per_split: dict = {}
    mags: dict = {}
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in table["clips"]:
            clip = build_clip(row)
            for r in clip.video_rows():
                w.writerow(r)
                rows += 1
            key = (clip.labels["split"], clip.labels["defect_family"])
            ent = per_split.setdefault(f"{key[0]}/{key[1]}", {
                "clips": 0, "frames": 0, "peak_tilt_req_deg": [],
                "median_tilt_req_deg": [], "peak_thrust_ratio": []})
            ent["clips"] += 1
            ent["frames"] += clip.labels["n_frames"]
            ent["peak_tilt_req_deg"].append(clip.labels["peak_tilt_req_deg"])
            ent["median_tilt_req_deg"].append(clip.labels["median_tilt_req_deg"])
            ent["peak_thrust_ratio"].append(clip.labels["peak_thrust_ratio"])
            if clip.labels["label"] == "infeasible":
                mags.setdefault(clip.labels["defect_family"], []).append(
                    clip.labels["magnitude_deg"])

    for ent in per_split.values():
        for k in ("peak_tilt_req_deg", "median_tilt_req_deg", "peak_thrust_ratio"):
            v = np.asarray(ent[k], dtype=float)
            ent[k] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
    summary = {
        "table_sha256": hashlib.sha256(Path(args.table).read_bytes()).hexdigest(),
        "n_clips": len(table["clips"]),
        "n_frames": rows,
        "by_split_and_defect": per_split,
        "magnitude_deg": {k: {"n": len(v), "min": float(np.min(v)),
                              "median": float(np.median(v)), "max": float(np.max(v))}
                          for k, v in mags.items()},
        "csv_bytes": OUT_CSV.stat().st_size,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"{OUT_CSV.relative_to(ROOT)}: {rows} rows, {OUT_CSV.stat().st_size / 1e6:.1f} MB")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n_clips", "n_frames", "magnitude_deg")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
