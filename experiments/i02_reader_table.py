"""Aggregate the reader's error on development clips by scene family and mount (Track I).

The per-clip scores on the volume (`i02/scores/<clip_id>_score.json`, produced by
`remote/i_score.py`) are collapsed here into the table the report and the downstream tracks
need: for feasible development clips only, the up-vector error and the depicted-on-required
lean gain, grouped by (family, mount). Held-out clips are excluded by construction -- the
manifest's split column is the filter, and a held-out score would have to exist already, which
`i_score.py` refuses without an explicit flag.

    python experiments/i02_reader_table.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/i02/manifest.csv"
SCORES = ROOT / "results/i02/scores"
OUT_CSV = ROOT / "results/i02/reader_feasible_grid.csv"
OUT_JSON = ROOT / "results/i02/reader_feasible_grid.json"


def _pull_scores(man: dict, split: str, defect: str) -> None:
    """Copy this split's score files from the volume, skipping ones already local."""
    import subprocess

    SCORES.mkdir(parents=True, exist_ok=True)
    pixi = Path.home() / "Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/modal"
    exe = str(pixi) if pixi.is_file() else "modal"
    cids = [cid for cid, m in man.items()
            if m.get("split") == split and (defect == "all" or m.get("defect_family") == defect)]
    missing = [c for c in sorted(cids) if not (SCORES / f"{c}_score.json").is_file()]
    print(f"pulling {len(missing)} score files ({len(cids) - len(missing)} already local)")
    for cid in missing:
        try:
            subprocess.run([exe, "volume", "get", "awmv-i-data",
                            f"i02/scores/{cid}_score.json", str(SCORES)],
                           check=True, capture_output=True, cwd=ROOT)
        except subprocess.CalledProcessError:
            pass                      # not scored yet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--defect", default="feasible",
                    help="defect_family to include ('feasible', 'all', or a family name)")
    ap.add_argument("--pull", action="store_true",
                    help="fetch score files for the selected clips from the Modal volume")
    args = ap.parse_args()

    man = {r["clip_id"]: r for r in csv.DictReader(MANIFEST.open())}
    if args.pull:
        _pull_scores(man, args.split, args.defect)
    rows = []
    for f in sorted(glob.glob(str(SCORES / "*_score.json"))):
        d = json.loads(Path(f).read_text())
        cid = Path(f).name.replace("_score.json", "")
        m = man.get(cid)
        if m is None or m["split"] != args.split:
            continue
        if args.defect != "all" and m["defect_family"] != args.defect:
            continue
        rows.append({"clip_id": cid, "family": m["family"], "uptilt": float(m["uptilt_deg"]),
                     "maneuver": m["maneuver"], "n": d["n_frames"],
                     "up_med": d["up_error_deg"]["median"], "up_p90": d["up_error_deg"]["p90"],
                     "gain_f": d["lean_slope_on_req"]["forward"],
                     "gain_l": d["lean_slope_on_req"]["left"]})
    if not rows:
        print("no scores match the filter")
        return 1

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    table = []
    for fam in sorted({r["family"] for r in rows}):
        for up in sorted({r["uptilt"] for r in rows}):
            cell = [r for r in rows if r["family"] == fam and r["uptilt"] == up]
            if not cell:
                continue
            up_med = np.array([r["up_med"] for r in cell])
            gains = np.array([r["gain_f"] for r in cell if r["gain_f"] is not None])
            table.append({
                "family": fam, "uptilt_deg": up, "clips": len(cell),
                "frames": int(sum(r["n"] for r in cell)),
                "up_med_deg": float(np.median(up_med)),
                "up_med_range": [float(up_med.min()), float(up_med.max())],
                "gain_f_med": float(np.median(gains)) if gains.size else None,
                "clip_ids": [r["clip_id"] for r in cell],
            })
    out = {"split": args.split, "defect": args.defect, "n_clips": len(rows),
           "n_frames": int(sum(r["n"] for r in rows)), "cells": table}
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"{OUT_CSV.relative_to(ROOT)}: {len(rows)} clips, {out['n_frames']} frames")
    print(f"{'family':14s} {'mount':>5s} {'clips':>5s} {'frames':>6s} {'up med':>7s} {'range':>15s} {'gain':>6s}")
    for c in table:
        g = c["gain_f_med"]
        print(f"{c['family']:14s} {c['uptilt_deg']:5.0f} {c['clips']:5d} {c['frames']:6d} "
              f"{c['up_med_deg']:7.2f} {c['up_med_range'][0]:6.2f}-{c['up_med_range'][1]:6.2f} "
              f"{(f'{g:.2f}' if g is not None else '   -'):>6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
