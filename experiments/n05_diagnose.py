"""Track N, N05 diagnostics: why does the level-start propagated arm underperform C2?

Post-hoc diagnostic (labelled as such; it does not change the pre-registered tables):
* the constant-offset claim. A start error is a fixed BODY rotation; C2 subtracts a 2-vector
  offset in the yaw-frame lean, which is not the same thing. The test isolates the start from
  DPVO drift by propagating with the TRUE rotations from the same level start.
* the coherence saturation. The scale-free correlation needs a varying required lean; many
  simulated manoeuvres hold a constant bank, where the correlation is undefined.
* an oracle-start arm, which shows what the tiers do when the start is right.

    PYTHONPATH=. python experiments/n05_diagnose.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.n05_sim_verdicts import ang_lean, tier_scores
from src.perception.frames import camera_to_body_pitched
from src.verdict.lean import depicted_lean_from_gravity
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n05"
E3 = np.array([0.0, 0.0, 1.0])
G0 = 9.80665


def inputs(cid, row):
    gt = np.load(ROOT / f"data/n05/gt/{cid}.npz")
    tgt = np.asarray(gt["t"], float)
    sc = np.load(ROOT / f"data/n05/scores/{cid}.npz")
    fidx = np.asarray(sc["frame"], int)
    t = fidx / 30.0
    lr = np.column_stack([np.interp(t, tgt, gt["lean_req"][:, k]) for k in range(2)])
    vel = np.column_stack([np.interp(t, tgt, gt["vel"][:, 0]), np.interp(t, tgt, gt["vel"][:, 1]),
                           np.interp(t, tgt, gt["yaw"])])
    c, s = np.cos(-vel[:, 2]), np.sin(-vel[:, 2])
    vreg = np.column_stack([c * vel[:, 0] / G0 + s * vel[:, 1] / G0,
                            -s * vel[:, 0] / G0 + c * vel[:, 1] / G0])
    R_cb = camera_to_body_pitched(np.deg2rad(float(row["mount"])))
    return gt, tgt, fidx, t, lr, vreg, R_cb


def truth_at(cid, rows, R_cb, t, tgt, gt):
    idx = np.clip(np.round(t * float(np.asarray(gt["gt_hz"]).ravel()[0])).astype(int),
                  0, len(np.asarray(gt["R_bw_cam"])) - 1)
    return np.einsum("nij,jk->nik", np.asarray(gt["R_bw_cam"], float)[idx], R_cb)


def main() -> int:
    rows = list(csv.DictReader(open(OUT / "clips.csv")))
    per = []
    for row in rows:
        cid = row["clip_id"]
        gt, tgt, fidx, t, lr, vreg, R_cb = inputs(cid, row)
        d = np.load(ROOT / f"data/n05/poses/sim200/scale1/{cid}.npz")
        R = np.asarray(d["c2w"], float)[:, :3, :3]
        Rtrue = truth_at(cid, rows, R_cb, t, tgt, gt)
        u0_level = R_cb.T @ E3
        u0_true = Rtrue[0].T @ E3
        up_dpvo = np.array([(R[k].T @ R[fidx[0]]) @ u0_level for k in fidx if k < len(R)])
        up_true_level = np.array([(Rtrue[k].T @ Rtrue[0]) @ u0_level for k in range(len(fidx))])
        up_oracle = np.array([(R[k].T @ R[fidx[0]]) @ u0_true for k in fidx if k < len(R)])
        n = min(len(up_true_level), len(lr))
        up_true_level = up_true_level[:n]
        up_oracle = up_oracle[:min(len(up_oracle), n)]
        l_true_level = depicted_lean_from_gravity(up_true_level, R_cb)
        l_oracle = depicted_lean_from_gravity(up_oracle, R_cb)
        sc = np.load(ROOT / f"data/n05/scores/{cid}.npz")
        l_reader = np.asarray(sc["lean_est"], float)[:n]
        ok = np.zeros(n, bool)
        ok[:len(l_true_level)] = True
        c2_true_level, r_true_level, _ = tier_scores(l_true_level, lr[:n], vreg[:n], ok)
        c2_oracle_full, r_oracle_full, _ = tier_scores(l_oracle, lr[:n], vreg[:n], ok)
        c2_reader, r_reader, _ = tier_scores(l_reader, lr[:n], vreg[:n], ok)
        c2_prop, r_prop, _ = tier_scores(depicted_lean_from_gravity(up_dpvo, R_cb), lr[:n],
                                         vreg[:n], ok[:len(up_dpvo)])
        std = float(np.max(np.std(lr[ok], axis=0))) if ok.sum() else float("nan")
        per.append({"clip_id": cid, "condition": row["condition"], "family": row["family"],
                    "mount": row["mount"], "req_std": std,
                    "C2_reader": c2_reader, "C2_prop_dpvo": c2_prop,
                    "C2_true_level": c2_true_level, "C2_oracle_start": c2_oracle_full,
                    "R_reader": r_reader, "R_prop": r_prop})
    with open(OUT / "diagnose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    out = {}
    for cond in ("feasible", "tilt_removed", "mirrored_roll", "tilt_scaled", "partial"):
        sub = [r for r in per if r["condition"] == cond]
        out[cond] = {k: float(np.nanmedian([r[k] for r in sub]))
                     for k in ("C2_reader", "C2_prop_dpvo", "C2_true_level", "C2_oracle_start",
                               "R_reader", "R_prop")}
        out[cond]["n"] = len(sub)
    # required-lean variability
    for cond in ("feasible", "tilt_removed", "mirrored_roll"):
        sub = [r["req_std"] for r in per if r["condition"] == cond]
        out[f"{cond}_req_std"] = {
            "frac_lt_0.001": float(np.mean(np.array(sub) < 1e-3)),
            "frac_lt_0.01": float(np.mean(np.array(sub) < 1e-2)),
            "median": float(np.nanmedian(sub))}
    # varying-lean subset: post-hoc coherence reading
    varying = [r for r in per if np.isfinite(r["req_std"]) and r["req_std"] >= 0.01]
    out["varying_subset"] = {
        "n": len(varying),
        "by_condition": {c: sum(r["condition"] == c for r in varying)
                         for c in ("feasible", "tilt_removed", "mirrored_roll", "tilt_scaled",
                                   "partial")},
        "median_R_reader_feasible": float(np.nanmedian([r["R_reader"] for r in varying
                                                        if r["condition"] == "feasible"])),
        "median_R_prop_feasible": float(np.nanmedian([r["R_prop"] for r in varying
                                                      if r["condition"] == "feasible"])),
    }
    # draw-to-draw comparison of C2_prop
    rep = [r for r in csv.DictReader(open(OUT / "per_clip_draw2.csv"))] if \
        (OUT / "per_clip_draw2.csv").exists() else []
    if rep:
        m = {r["clip_id"]: r for r in rep}
        diffs = [abs(float(r["C2_prop_dpvo"]) - float(m[r["clip_id"]]["C2_propagated"]))
                 for r in per if r["clip_id"] in m
                 and np.isfinite(float(r["C2_prop_dpvo"])) and np.isfinite(float(m[r["clip_id"]]["C2_propagated"]))]
        out["draw_difference_C2_prop"] = {"n": len(diffs),
                                         "median_abs": float(np.median(diffs)),
                                         "p90_abs": float(np.percentile(diffs, 90))}
    (OUT / "diagnose.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
