"""Track G3: the auxiliary yaw-control slice, scored against its own null.

The slice re-renders 12 feasible dev clips with the camera yawed about world up by
``alpha_i = min(tilt_req_deg_i, 60)``.  The geometry is unchanged: the schema lean is
invariant to a world yaw (``tests/test_g3_yaw_invariance.py``), so any response of the window
statistic comes from the reader looking at a different part of the scene.  The null is the same
clip's ordinary stride-1 score, which is the alpha = 0 render.

Reported: the primary arm's flag rate on the slice against the null, and the reader's yaw-induced
change in lean (median, p90, max) against G01's real-clip 4.75 deg median shift.

    pixi run python experiments/g3_yaw.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.g3_ablation import (ALPHA, ARMS, arm_mask, arm_value,          # noqa: E402
                                     quantile)
from experiments.g3_data import load_clip, read_manifest                       # noqa: E402
from src.verdict.segments import window_statistics                            # noqa: E402

PRIMARY = next(a for a in ARMS if a["name"] == "coh12_p1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/g06/scores")
    ap.add_argument("--yaw-scores", default="results/g06/yaw_scores")
    ap.add_argument("--config", default="configs/g06_yaw_slice.json")
    ap.add_argument("--manifest",
                    default="/home/kevin/Documents/Projects/awmv-track-i/results/i02/manifest.csv")
    ap.add_argument("--out", default="results/g06/yaw_slice")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = read_manifest(args.manifest)
    cfg = json.loads(Path(args.config).read_text())
    ids = [c["clip_id"] for c in cfg["clips"]]

    # thresholds from the dev feasible clips (the 12 are all feasible dev clips, so the
    # leave-one-scene-out fold that held their scene out is used, exactly as in A1)
    all_ids = [p.name.replace("_score.json", "")
               for p in Path(args.scores).glob("*_score.json")]
    dev = {c: load_clip(c, args.scores, man) for c in all_ids}
    feasible = [c for c in dev if man.get(c, {}).get("label") == "feasible"]
    stats = {c: window_statistics(dev[c], length=PRIMARY["length"], hop=4) for c in dev}
    vals = {c: arm_value(stats[c], PRIMARY) for c in dev}

    rows, shifts = [], []
    flagged = {"null": 0, "yaw": 0}
    testable = {"null": 0, "yaw": 0}
    for cid in ids:
        row = man[cid]
        seed = int(row["scene_seed"])
        fit = [c for c in feasible if int(man[c]["scene_seed"]) != seed]
        tau = quantile([vals[c] for c in fit], ALPHA)
        for tag, sdir in (("null", args.scores), ("yaw", args.yaw_scores)):
            f = Path(sdir) / f"{cid}_score.json"
            if not f.is_file():
                continue
            x = load_clip(cid, sdir, man)
            st = window_statistics(x, length=PRIMARY["length"], hop=4)
            v = arm_value(st, PRIMARY)
            if not (np.isfinite(v) and np.isfinite(tau)):
                continue
            testable[tag] += 1
            flagged[tag] += int(arm_mask(st, PRIMARY, tau).any())
        # lean shift between the yawed render and the null at the same frames
        fn = Path(args.scores) / f"{cid}_score.json"
        fy = Path(args.yaw_scores) / f"{cid}_score.json"
        if fn.is_file() and fy.is_file():
            a = {int(r["frame"]): r for r in json.loads(fn.read_text())["rows"]}
            b = {int(r["frame"]): r for r in json.loads(fy.read_text())["rows"]}
            common = sorted(set(a) & set(b))
            if common:
                dl = np.array([[b[i]["lean_est_x"] - a[i]["lean_est_x"],
                                b[i]["lean_est_y"] - a[i]["lean_est_y"]] for i in common])
                ang = np.degrees(np.arctan2(dl[:, 1], dl[:, 0]))
                mag = np.degrees(np.arctan2(np.linalg.norm(dl, axis=1), 1.0))
                shifts.append({"clip_id": cid, "family": row["family"],
                               "uptilt_deg": float(row["uptilt_deg"]), "n": len(common),
                               "median_lean_shift_deg": float(np.median(mag)),
                               "p90": float(np.percentile(mag, 90)),
                               "max": float(np.max(mag)),
                               "median_azimuth_deg": float(np.median(ang))})
        rows.append({"clip_id": cid, "family": row["family"],
                     "uptilt_deg": float(row["uptilt_deg"]),
                     "peak_tilt_req_deg": float(row["peak_tilt_req_deg"]),
                     "tau": tau, "null_stat": vals.get(cid)})

    summary = {
        "n_clips": len(ids),
        "testable": testable, "flagged": flagged,
        "false_flag_rate": {k: (flagged[k] / testable[k] if testable[k] else None)
                            for k in flagged},
        "lean_shift_deg": {
            "median_of_clip_medians": float(np.median([s["median_lean_shift_deg"] for s in shifts]))
            if shifts else None,
            "p90_of_clip_p90": float(np.percentile([s["p90"] for s in shifts], 90)) if shifts else None,
            "max": float(max(s["max"] for s in shifts)) if shifts else None,
            "real_clip_reference_deg": 4.75,
        },
        "per_clip": rows, "shifts": shifts,
    }
    (out / "yaw_summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# Auxiliary yaw slice", "",
             f"clips: {len(ids)}; testable null {testable['null']}, yaw {testable['yaw']}",
             f"flagged: null {flagged['null']}, yaw {flagged['yaw']}",
             f"reader yaw-induced lean shift: median {summary['lean_shift_deg']['median_of_clip_medians']}, "
             f"p90 {summary['lean_shift_deg']['p90_of_clip_p90']}, max {summary['lean_shift_deg']['max']} deg"]
    (out / "yaw_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
