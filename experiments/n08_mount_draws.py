"""Track N2, N08: Part B(iii) -- two-draw robust aggregation against the high-mount drift.

Pre-registration ``configs/n08_mount_draws_prereg.yaml`` (committed before any score here).

CPU reanalysis of the two stored DPVO draws; no GPU job.

    pixi run python experiments/n08_mount_draws.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.n06_mount_drift import true_attitude

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n2/n08"
DRAW1 = ROOT / "data/n05/poses/sim200/scale1"
DRAW2 = ROOT / "data/n05/poses/sim200/scale1_rep2"
E3 = np.array([0.0, 0.0, 1.0])
H = 10.0
MOUNTS = (0.0, 15.0, 30.0, 45.0)


def up_of(R):
    return np.einsum("...ji,j->...i", np.asarray(R, float), E3)


def ang(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def horizon_pairs(t, h=H):
    """N06's 10 s pairing: for every sample i, the sample j nearest t_i + h (within 2/15 s)."""
    t = np.asarray(t, float)
    n = len(t)
    idx = np.searchsorted(t, t + h)
    j = np.minimum(idx, n - 1)
    ok = (idx < n) & (j > np.arange(n)) & (np.abs(t[j] - t - h) <= 2.0 / 15.0)
    i = np.arange(n)[ok]
    return i, j[ok]


def preds(R, t, Rt):
    """Estimated world-up at each horizon sample, N06's construction."""
    i, j = horizon_pairs(t)
    ut = up_of(Rt)
    rel = np.einsum("nji,njk->nik", np.asarray(R, float)[j], np.asarray(R, float)[i])
    return np.einsum("nij,nj->ni", rel, ut[i]), ut[j], i, j


def errs(p, utj):
    return ang(p, utj)


def summarise(vals, rows, key, cond):
    out = {}
    for m in MOUNTS:
        v = [x for x, r in zip(vals, rows) if float(r["mount"]) == float(m) and cond(r)
             and np.isfinite(x)]
        out[str(m)] = {"n": len(v), "median": float(np.median(v)) if v else None,
                       "max": float(np.max(v)) if v else None}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / "results/n05/clips.csv"))
            if r["condition"] == "feasible"]
    per = []
    for n, row in enumerate(rows):
        cid = row["clip_id"]
        d1 = np.load(DRAW1 / f"{cid}.npz")
        d2 = np.load(DRAW2 / f"{cid}.npz")
        R1 = np.asarray(d1["c2w"], float)[:, :3, :3]
        R2 = np.asarray(d2["c2w"], float)[:, :3, :3]
        t1 = np.arange(int(d1["n_pose"])) / float(d1["fps"])
        t2 = np.arange(int(d2["n_pose"])) / float(d2["fps"])
        # a clip is aggregable only if both draws share the horizon sample grid; the inputs are
        # identical so this holds, but check rather than assume
        if len(t1) != len(t2) or not np.allclose(t1, t2):
            per.append({"clip_id": cid, "mount": row["mount"], "family": row["family"],
                        "aggregable": False})
            continue
        Rt = true_attitude(cid, t1)
        p1, utj, i, j = preds(R1, t1, Rt)
        p2, _, _, _ = preds(R2, t1, Rt)
        u1 = p1 / np.linalg.norm(p1, axis=1, keepdims=True)
        u2 = p2 / np.linalg.norm(p2, axis=1, keepdims=True)
        dis = ang(u1, u2)
        agg = u1 + u2
        agg = agg / np.linalg.norm(agg, axis=1, keepdims=True)
        e1, e2, ea = errs(u1, utj), errs(u2, utj), errs(agg, utj)
        fail = bool(max(e1.max(), e2.max()) >= 85.0)
        per.append({
            "clip_id": cid, "mount": row["mount"], "family": row["family"],
            "aggregable": True, "n_samples": int(len(e1)),
            "drift_d1": float(np.median(e1)), "drift_d2": float(np.median(e2)),
            "drift_agg": float(np.median(ea)),
            "max_d1": float(e1.max()), "max_d2": float(e2.max()), "max_agg": float(ea.max()),
            "disagreement_median": float(np.median(dis)),
            "disagreement_p90": float(np.percentile(dis, 90)),
            "disagreement_max": float(dis.max()),
            "frac_dis_gt_15": float(np.mean(dis > 15.0)),
            "fail_flag": fail,
        })
        if n % 20 == 0:
            print("clip", n, flush=True)
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(per)

    good = [r for r in per if r.get("aggregable") and not r.get("fail_flag")]
    flags = [r for r in per if r.get("aggregable") and r.get("fail_flag")]
    forest = lambda r: r["family"] == "forest_ridge"
    summary = {"n_feasible": len(rows), "n_aggregable": len(good) + len(flags),
               "n_fail_flagged": len(flags),
               "fail_flagged_clips": [r["clip_id"] for r in flags],
               "aggregation": "componentwise mean of the two unit predicted-ups, renormalised"}
    for tag, fn in (("all", lambda r: True), ("forest", forest)):
        sub = [r for r in good if fn(r)]
        summary[tag] = {}
        for k in ("drift_d1", "drift_d2", "drift_agg", "max_d1", "max_d2", "max_agg",
                  "disagreement_median", "disagreement_p90", "frac_dis_gt_15"):
            vals = [r[k] for r in sub]
            summary[tag][k] = summarise(vals, sub, k, lambda r: True)
        means = [(r["drift_d1"] + r["drift_d2"]) / 2.0 for r in sub]
        summary[tag]["mean_of_singles"] = summarise(means, sub, "m", lambda r: True)
    # criteria
    def med(tag, mount, key):
        v = summary[tag][key][str(mount)]["median"]
        return v
    f45_agg = med("forest", 45.0, "drift_agg")
    f45_single = (med("forest", 45.0, "drift_d1") + med("forest", 45.0, "drift_d2")) / 2.0
    f30_agg = med("forest", 30.0, "drift_agg")
    f30_best = min(med("forest", 30.0, "drift_d1"), med("forest", 30.0, "drift_d2"))
    f45_max_agg = summary["forest"]["max_agg"]["45.0"]["max"]
    f45_max_single = max(summary["forest"]["max_d1"]["45.0"]["max"],
                         summary["forest"]["max_d2"]["45.0"]["max"])
    c1 = (f45_agg is not None and f45_single and f45_agg <= 0.75 * f45_single
          and f30_agg is not None and f30_best and f30_agg <= 1.10 * f30_best)
    c2 = (f45_max_agg is not None and f45_max_single is not None
          and f45_max_agg <= f45_max_single)
    d45 = med("forest", 45.0, "disagreement_median")
    summary["criteria"] = {
        "c1_reduction": bool(c1),
        "c1_numbers": {"forest_45_aggregate": f45_agg, "forest_45_mean_of_singles": f45_single,
                       "forest_30_aggregate": f30_agg, "forest_30_better_single": f30_best},
        "c2_no_harm": bool(c2),
        "c2_numbers": {"forest_45_max_aggregate": f45_max_agg,
                       "forest_45_max_better_single": f45_max_single},
        "c3_forest_45_median_disagreement": d45,
        "reading": ("a common bias dominates: averaging does not help" if not c1
                    else "draw variance dominates: averaging helps"),
    }
    summary["note"] = ("10 s oracle-start drift, N06 statistic: median over horizon samples per "
                       "clip, then median across clips, by mount; forest is 26 clips, 6-7 per "
                       "mount, so every cell is descriptive. Fail-flagged clips are dropped from "
                       "every arm. Draw 1 = scale1, draw 2 = scale1_rep2.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["criteria"], indent=1))
    for tag in ("all", "forest"):
        print(tag, "agg by mount",
              {m: summary[tag]["drift_agg"][m]["median"] for m in ("0.0", "15.0", "30.0", "45.0")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
