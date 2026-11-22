"""Track G3: A1 window-statistic ablation on the simulated dev set.

Arms are the pre-registered ones in ``configs/g06_segments_prereg.yaml``: the 12-frame
coherence scan (primary), the 8- and 12-frame residual scans, sign, persistence 2, and the
whole-clip median baseline.  Thresholds are leave-one-scene-seed-out conformal 95th percentiles
on the feasible dev clips; a clip uses the threshold of the fold that held its scene out.

Nothing is tuned here.  Outputs go to ``results/g06/ablation``.

    pixi run python experiments/g3_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.g3_data import load_split, read_manifest                     # noqa: E402
from src.verdict.segments import (clip_median_whole, clip_max,                # noqa: E402
                                  clip_max_persistent, flagged_windows,
                                  flagged_windows_persistent, interval_scores,
                                  merge_intervals, window_statistics)

ARMS = [
    {"name": "coh12_p1", "tier": "coherence", "length": 12, "hop": 4, "persistence": 1},
    {"name": "coh12_p2", "tier": "coherence", "length": 12, "hop": 4, "persistence": 2},
    {"name": "coh8_p1", "tier": "coherence", "length": 8, "hop": 4, "persistence": 1},
    {"name": "res12_p1", "tier": "residual", "length": 12, "hop": 4, "persistence": 1},
    {"name": "res8_p1", "tier": "residual", "length": 8, "hop": 4, "persistence": 1},
    {"name": "res8_p2", "tier": "residual", "length": 8, "hop": 4, "persistence": 2},
    {"name": "sign12_p1", "tier": "sign", "length": 12, "hop": 4, "persistence": 1},
]
BASE = {"name": "median_baseline", "tier": "median", "length": 0, "hop": 0, "persistence": 1}
ALPHA = 0.05


def arm_value(stats, arm) -> float:
    if arm["tier"] == "median":
        return float("nan")
    if arm["persistence"] > 1:
        return clip_max_persistent(stats, arm["tier"], arm["persistence"])
    return clip_max(stats, arm["tier"])


def arm_mask(stats, arm, tau) -> np.ndarray:
    if arm["persistence"] > 1:
        return flagged_windows_persistent(stats, arm["tier"], tau, arm["persistence"])
    return flagged_windows(stats, arm["tier"], tau)


def quantile(vals, alpha=ALPHA) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.quantile(v, 1.0 - alpha)) if len(v) >= 8 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/g06/scores")
    ap.add_argument("--manifest",
                    default="/home/kevin/Documents/Projects/awmv-track-i/results/i02/manifest.csv")
    ap.add_argument("--out", default="results/g06/ablation")
    ap.add_argument("--yaw-scores", default="results/g06/yaw_scores")
    ap.add_argument("--yaw-config", default="configs/g06_yaw_slice.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = read_manifest(args.manifest)
    clips = load_split(args.scores, man)
    dev = {c: x for c, x in clips.items() if man.get(c, {}).get("split") == "dev"}
    print(f"{len(dev)} dev clips with scores")

    # window statistics per length, once
    stats = {}
    for length in (8, 12, 16):
        for c, x in dev.items():
            stats[(c, length)] = window_statistics(x, length=length, hop=4)

    values = {a["name"]: {} for a in ARMS}
    values[BASE["name"]] = {}
    for c, x in dev.items():
        values[BASE["name"]][c] = clip_median_whole(x)
        for a in ARMS:
            values[a["name"]][c] = arm_value(stats[(c, a["length"])], a)

    seeds = sorted({int(man[c]["scene_seed"]) for c in dev})
    feasible = [c for c in dev if man[c]["label"] == "feasible"]

    thresholds = {}
    for a in ARMS + [BASE]:
        per_fold = {}
        for s in seeds:
            fit = [c for c in feasible if int(man[c]["scene_seed"]) != s]
            per_fold[s] = quantile([values[a["name"]][c] for c in fit])
        thresholds[a["name"]] = {
            "per_fold": per_fold,
            "all_dev": quantile([values[a["name"]][c] for c in feasible]),
            "spread": [min(per_fold.values()), max(per_fold.values())]}

    def tau_for(arm_name, clip_id):
        return thresholds[arm_name]["per_fold"][int(man[clip_id]["scene_seed"])]

    # ---- per-clip decisions and segments
    dec = defaultdict(dict)
    for a in ARMS + [BASE]:
        for c, x in dev.items():
            tau = tau_for(a["name"], c)
            stat = values[a["name"]][c]
            if a["name"] == BASE["name"]:
                mask = np.zeros(len(x), dtype=bool)
                if np.isfinite(tau) and np.isfinite(stat) and stat > tau:
                    mask[:] = True
            else:
                mask = (arm_mask(stats[(c, a["length"])], a, tau)
                        if np.isfinite(tau) else np.zeros(len(x), dtype=bool))
            sc = interval_scores(mask, x.truth)
            sc["flagged"] = bool(mask.any())
            sc["testable"] = bool(np.isfinite(stat) and np.isfinite(tau))
            sc["stat"] = float(stat)
            sc["tau"] = float(tau)
            nover = len(merge_intervals(mask))
            if x.truth.any():
                nover_true = len(merge_intervals(mask & x.truth))
                sc["false_segments"] = max(0, nover - nover_true)
            else:
                sc["false_segments"] = nover
            dec[a["name"]][c] = sc

    # ---- aggregate tables
    det_rows, frr_rows = [], []
    for a in ARMS + [BASE]:
        for fam in ("urban_checker", "forest_ridge", "industrial"):
            for tgt in ("tilt_removed", "mirrored_roll", "tilt_scaled", "partial"):
                for frac in ("0.1", "0.25", "0.5", "1.0"):
                    ids = [c for c in dev if man[c]["family"] == fam
                           and man[c]["defect_family"] == tgt
                           and (not x_frac(man[c]) or abs(x_frac(man[c]) - float(frac)) < 1e-6)]
                    if not ids:
                        continue
                    test = [c for c in ids if dec[a["name"]][c]["testable"]]
                    k = sum(dec[a["name"]][c]["flagged"] for c in test)
                    iou = [dec[a["name"]][c]["iou"] if dec[a["name"]][c]["flagged"] else 0.0
                           for c in test]
                    iou_c = [dec[a["name"]][c]["iou"] for c in test
                             if dec[a["name"]][c]["flagged"]]
                    f1 = [dec[a["name"]][c]["f1"] if dec[a["name"]][c]["flagged"] else 0.0
                          for c in test]
                    onset = [dec[a["name"]][c]["onset_err_frames"] for c in test
                             if dec[a["name"]][c]["flagged"]]
                    det_rows.append({
                        "arm": a["name"], "family": fam, "defect": tgt, "fraction": frac,
                        "n_testable": len(test), "n_clips": len(ids), "k": k,
                        "rate": k / len(test) if test else None,
                        "iou_med": float(np.nanmedian(iou)) if iou else None,
                        "iou_med_detected": float(np.nanmedian(iou_c)) if iou_c else None,
                        "f1_med": float(np.nanmedian(f1)) if f1 else None,
                        "onset_med": float(np.nanmedian(onset)) if onset else None,
                    })
        # false-reject / controls
        for fam in ("urban_checker", "forest_ridge", "industrial", "ALL"):
            ids = [c for c in feasible if fam == "ALL" or man[c]["family"] == fam]
            test = [c for c in ids if dec[a["name"]][c]["testable"]]
            k = sum(dec[a["name"]][c]["flagged"] for c in test)
            mins = sum(len(dev[c]) for c in test) / 30.0 / 60.0
            fseg = sum(dec[a["name"]][c]["false_segments"] for c in test)
            frr_rows.append({"arm": a["name"], "family": fam, "n_testable": len(test),
                             "k": k, "rate": k / len(test) if test else None,
                             "false_segments": fseg, "minutes": mins,
                             "false_segments_per_min": fseg / mins if mins else None})

    def dump(rows, name):
        if not rows:
            return
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    dump(det_rows, "detection.csv")
    dump(frr_rows, "false_reject.csv")
    summary = {"n_dev_clips": len(dev), "n_feasible": len(feasible), "seeds": seeds,
               "alpha": ALPHA, "thresholds": thresholds,
               "decisions": {a: {c: {k: v for k, v in s.items() if k != "segments"}
                                 for c, s in dec[a].items()} for a in dec}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- console: primary arm versus baseline and comparison, pooled partial cells
    print("\narm / family: FRR (k/n testable)")
    for r in frr_rows:
        if r["rate"] is not None:
            print(f"  {r['arm']:18s} {r['family']:14s} {r['k']:3d}/{r['n_testable']:3d} "
                  f"{100*r['rate']:5.1f}%  fseg/min {r['false_segments_per_min']:.2f}")
    print("\npartial defects (family pooled), 25% coverage:")
    for a in ARMS:
        rows = [r for r in det_rows if r["arm"] == a["name"] and r["fraction"] == "0.25"]
        n = sum(r["n_testable"] for r in rows)
        k = sum(r["k"] for r in rows)
        if n:
            print(f"  {a['name']:18s} {k:3d}/{n:3d}  {100*k/n:5.1f}%")
    return 0


def x_frac(row: dict) -> float:
    v = row.get("interval_fraction") or 0.0
    return float(v or 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
