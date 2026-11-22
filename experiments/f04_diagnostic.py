"""Track F2, part 1 diagnostic: does the fitted path scale collapse?

Reads the combined summary's per-clip rows (via `results/f04_spline/rows.csv`) and part 2's
Sim(3) scales, and writes `results/f04_spline/diagnostic.json`: per primary clip the real
statistic, the fitted scale, the ratio to the Sim(3) truth scale, and per-path medians. This is
the measurement behind F04's degeneracy section.

    python experiments/f04_diagnostic.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PRIMARY = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_9",
           "indoor_forward_10")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/f04_spline/rows.csv")
    ap.add_argument("--lean", default="results/f05_lean_error/lean_error.json")
    ap.add_argument("--out", default="results/f04_spline/diagnostic.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.rows)))
    lean = json.loads(Path(args.lean).read_text())
    sim3 = {p: {s: v["scale_sim3"] for s, v in b.items()}
            for p, b in lean["paths"].items()}
    out = {"clips": [], "by_path": {}}
    for path_name in sorted({r["extractor"] for r in rows}):
        sub = [r for r in rows if r["extractor"] == path_name and r["path"] == "pixel"
               and r["cond"] == "real"
               and r["arm"] == "estimated_attitude" and r["full_path"] == "True"
               and r["seq"] in PRIMARY]
        clips = []
        for r in sub:
            s = float(r["scale"])
            s3 = sim3[path_name][r["seq"]]
            clips.append({"clip": r["clip"], "rho": float(r["rho"]),
                          "rho_nov0": float(r["rho_nov0"]),
                          "scale": s, "sim3_scale": s3, "scale_ratio": s / s3,
                          "knots": int(float(r["knots"]))})
        if not clips:
            continue
        vals = [c["rho"] for c in clips]
        ratios = [c["scale_ratio"] for c in clips]
        out["clips"].extend([{"path": path_name, **c} for c in clips])
        out["by_path"][path_name] = {
            "n": len(clips), "median_rho": float(np.median(vals)),
            "max_rho": float(max(vals)),
            "median_scale_ratio": float(np.median(ratios)),
            "n_scale_below_half": int(sum(1 for x in ratios if x < 0.5))}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_path"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
