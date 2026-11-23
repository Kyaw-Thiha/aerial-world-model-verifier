"""Track N2, N07: F0 + sign + C2 + R tiers on the propagated lean over the 200 dev sim clips.

Pre-registration ``configs/n07_sim_tiers_prereg.yaml`` (committed before any score here).

CPU reanalysis of the N05 reader score cache and the two stored DPVO pose draws. No GPU job.

    pixi run python experiments/n07_sim_tiers.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.n05_sim_verdicts import ang_lean, tier_scores
from src.depicted.clips import up_body_from_lean
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import angle_between_deg
from src.verdict.lean import depicted_lean_from_gravity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n2/n07"
GT = ROOT / "data/n05/gt"
SCORES = ROOT / "data/n05/scores"
DRAW1 = ROOT / "data/n05/poses/sim200/scale1"
DRAW2 = ROOT / "data/n05/poses/sim200/scale1_rep2"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"
E3 = np.array([0.0, 0.0, 1.0])
G0 = 9.80665
SIGMA_N_DEG = 1.910587818373582
EPS_SIGN = float(np.sin(np.deg2rad(SIGMA_N_DEG)))
WIN, HOP, MIN_USABLE, MIN_SIGN = 8, 4, 6, 2
ARMS = ("reader", "dpvo_level", "dpvo_reader_start", "dpvo_oracle", "dpvo_oracle_2deg")
TIERS = ("F0", "C2", "R", "S")


# ------------------------------------------------------------------------------- statistics
def f0_stat(l_dep, l_req, ok):
    """K03 fixed-calibration thrust-axis angle: clip median, no nuisance."""
    if ok.sum() < 8:
        return float("nan")
    a = up_body_from_lean(np.asarray(l_dep, float)[ok])
    b = up_body_from_lean(np.asarray(l_req, float)[ok])
    good = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if good.sum() < 8:
        return float("nan")
    return float(np.median(angle_between_deg(a[good], b[good])))


def _windows(n):
    if n <= WIN:
        return [(0, n)]
    return [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]


def _sign_agreement(a, b, eps):
    keep = (np.abs(a) > eps) & (np.abs(b) > eps)
    if keep.sum() == 0:
        return 0.5, 0
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep]))), int(keep.sum())


def sign_stat(dep, req):
    """M04 db1.0: max over published windows of 1 - min(s_x, s_y); NaN if no usable window."""
    vals = []
    for lo, hi in _windows(len(dep)):
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < MIN_USABLE:
            continue
        ss, ns = [], []
        for j in range(2):
            s, n = _sign_agreement(d[:, j], q[:, j], EPS_SIGN)
            ss.append(s); ns.append(n)
        if min(ns) < MIN_SIGN:
            continue
        vals.append(1.0 - min(ss))
    return float(max(vals)) if vals else float("nan")


def all_tiers(l_dep, l_req, vreg, ok):
    c2, r, _ = tier_scores(l_dep, l_req, vreg, ok)
    return {"F0": f0_stat(l_dep, l_req, ok), "C2": c2, "R": r, "S": sign_stat(l_dep, l_req)}


# ------------------------------------------------------------------------------- per clip
def true_c2w(gt, R_cb, t):
    idx = np.clip(np.round(np.asarray(t) * float(np.asarray(gt["gt_hz"]).ravel()[0])).astype(int),
                  0, len(np.asarray(gt["R_bw_cam"])) - 1)
    return np.einsum("nij,jk->nik", np.asarray(gt["R_bw_cam"], float)[idx], R_cb)


def propagate(R, fidx, u0):
    """up_k = (R_k^T R_0) u0 in camera coordinates; NaN where the pose is missing."""
    out = np.full((len(fidx), 3), np.nan)
    good = fidx < len(R)
    n = int(good.sum())
    if n:
        out[:n] = np.einsum("nij,j->ni", np.einsum("nij->nji", R[fidx[:n]]), R[fidx[0]] @ u0)
    return out


def tangent_frame(u):
    t = np.cross(u, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(u, np.array([1.0, 0.0, 0.0]))
    t = t / np.linalg.norm(t)
    b = np.cross(u, t)
    return t, b


def build(row, draw_dir, rng, sens=True):
    cid = row["clip_id"]
    gt = np.load(GT / f"{cid}.npz")
    tgt = np.asarray(gt["t"], float)
    sc = np.load(SCORES / f"{cid}.npz")
    fidx = np.asarray(sc["frame"], int)
    t = fidx / 30.0
    lr = np.column_stack([np.interp(t, tgt, np.asarray(gt["lean_req"], float)[:, k])
                          for k in range(2)])
    vel = np.column_stack([np.interp(t, tgt, np.asarray(gt["vel"], float)[:, 0]),
                           np.interp(t, tgt, np.asarray(gt["vel"], float)[:, 1]),
                           np.interp(t, tgt, np.asarray(gt["yaw"], float))])
    c, s = np.cos(-vel[:, 2]), np.sin(-vel[:, 2])
    vreg = np.column_stack([c * vel[:, 0] / G0 + s * vel[:, 1] / G0,
                            -s * vel[:, 0] / G0 + c * vel[:, 1] / G0])
    R_cb = camera_to_body_pitched(np.deg2rad(float(row["mount"])))
    assert np.max(np.abs(R_cb - np.asarray(gt["R_cb"], float))) < 1e-9

    R = np.asarray(np.load(draw_dir / f"{cid}.npz")["c2w"], float)[:, :3, :3]
    l_reader = np.asarray(sc["lean_est"], float)
    up0_reader = np.asarray(sc["up_est"], float)[0]
    up0_reader = up0_reader / np.linalg.norm(up0_reader)
    Rtrue = true_c2w(gt, R_cb, t)
    up0_true = Rtrue[0].T @ E3

    ups = {
        "dpvo_level": propagate(R, fidx, R_cb.T @ E3),
        "dpvo_reader_start": propagate(R, fidx, up0_reader),
        "dpvo_oracle": propagate(R, fidx, up0_true),
    }
    leans = {"reader": l_reader}
    for k, u in ups.items():
        leans[k] = depicted_lean_from_gravity(u, R_cb)

    # 2-degree start sensitivity: 8 draws, the tier statistic is the median over draws
    per_draw = defaultdict(list)
    if sens:
        tang, binorm = tangent_frame(up0_true)
        th = np.deg2rad(2.0)
        for _ in range(8):
            phi = rng.uniform(0.0, 2.0 * np.pi)
            axis = np.cos(phi) * tang + np.sin(phi) * binorm
            u0 = np.cos(th) * up0_true + np.sin(th) * axis
            u0 = u0 / np.linalg.norm(u0)
            l = depicted_lean_from_gravity(propagate(R, fidx, u0), R_cb)
            ok2 = np.isfinite(l).all(1) & np.isfinite(lr).all(1) & np.isfinite(vreg).all(1)
            st = all_tiers(l, lr, vreg, ok2)
            for tier in TIERS:
                per_draw[tier].append(st[tier])

    out = {"clip_id": cid, "family": row["family"], "scene_seed": row["scene_seed"],
           "mount": float(row["mount"]), "condition": row["condition"],
           "factor": row["factor"], "interval_fraction": row["interval_fraction"]}
    for arm, l in leans.items():
        ok = np.isfinite(l).all(1) & np.isfinite(lr).all(1) & np.isfinite(vreg).all(1)
        st = all_tiers(l, lr, vreg, ok)
        for tier in TIERS:
            out[f"{tier}|{arm}"] = st[tier]
        out[f"n|{arm}"] = int(ok.sum())
    for tier in TIERS:
        v = np.asarray(per_draw[tier], float)
        out[f"{tier}|dpvo_oracle_2deg"] = (float(np.nanmedian(v)) if v.size and np.isfinite(v).any()
                                           else float("nan"))
    ok = np.isfinite(leans["dpvo_reader_start"]).all(1) & np.isfinite(lr).all(1)
    out["n|dpvo_oracle_2deg"] = int(ok.sum())
    # peak required tilt (M05 dynamic definition), from the manifest-independent gt side
    out["peak_tilt_req_deg"] = float(np.nanmax(np.degrees(np.arcsin(
        np.clip(np.linalg.norm(np.asarray(gt["lean_req"], float), axis=1), 0, 1)))))
    return out


# ------------------------------------------------------------------------------- thresholds
def _vals_and_flags(rows, tier, arm):
    feas = [r for r in rows if r["condition"] == "feasible"]
    flags = {}
    for r in rows:
        others = [f[f"{tier}|{arm}"] for f in feas
                  if f["scene_seed"] != r["scene_seed"]]
        others = [v for v in others if np.isfinite(v)]
        thr = float(np.quantile(others, 0.95)) if len(others) >= 5 else np.nan
        flags[r["clip_id"]] = bool(np.isfinite(r[f"{tier}|{arm}"]) and np.isfinite(thr)
                                   and r[f"{tier}|{arm}"] > thr)
    return flags


def cut_flags(rows, tier, arm, k):
    """Exactly k pooled feasible clips flagged (cut between the k-th and (k+1)-th largest)."""
    feas = [r for r in rows if r["condition"] == "feasible"]
    vals = sorted([f[f"{tier}|{arm}"] for f in feas if np.isfinite(f[f"{tier}|{arm}"])],
                  reverse=True)
    if not vals:
        return {}, float("nan")
    kk = min(k, len(vals))
    thr = vals[kk] if kk < len(vals) else vals[-1]
    flags = {r["clip_id"]: bool(np.isfinite(r[f"{tier}|{arm}"])
                                and r[f"{tier}|{arm}"] > thr) for r in rows}
    return flags, float(thr)


def rate(sub, flags):
    v = [flags[r["clip_id"]] for r in sub]
    return f"{int(sum(v))}/{len(v)}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "results/n05/clips.csv")))
    assert len(rows) == 200 and all(r["clip_id"].startswith("i02_dev") for r in rows)
    # dynamic-clip flag from the Track I manifest (read-only), joined on clip_id
    dyn = {r["clip_id"]: float(r["peak_tilt_req_deg"]) for r in csv.DictReader(open(MANIFEST))}
    for r in rows:
        r["dynamic"] = dyn.get(r["clip_id"], float("nan")) >= 15.0

    rng = np.random.default_rng(1)
    per = []
    for i, r in enumerate(rows):
        rec1 = build(r, DRAW1, rng)
        rec1["draw"] = 1
        rec2 = build(r, DRAW2, rng, sens=False)
        rec2["draw"] = 2
        per.append(rec1)
        # draw-2 rows carry only the two robustness arms' tiers
        rec2["clip_id"] = r["clip_id"]
        per.append(rec2)
        if i % 25 == 0:
            print("clip", i, flush=True)

    # per-clip CSV: one row per clip per draw
    keys = [k for k in per[0] if not k.startswith(("F0|", "C2|", "R|", "S|"))]
    keys += [f"{t}|{a}" for t in TIERS for a in ARMS]
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(per)

    d1 = [r for r in per if r["draw"] == 1]
    d2 = [r for r in per if r["draw"] == 2]
    summary = {"n": len(d1), "tiers": {}, "k_ref": {}, "thresholds": {}}
    conds = ("feasible", "tilt_removed", "mirrored_roll", "tilt_scaled", "partial",
             "thrust_limit")
    families = ("urban_checker", "forest_ridge", "industrial")
    mounts = (0.0, 15.0, 30.0, 45.0)

    # natural thresholds on draw 1, and per-tier k_ref from the reader arm
    nat = {}
    for tier in TIERS:
        for arm in ARMS:
            nat[(tier, arm)] = _vals_and_flags(d1, tier, arm)
        nat_reader = sum(nat[(tier, "reader")][r["clip_id"]]
                         for r in d1 if r["condition"] == "feasible")
        summary["k_ref"][tier] = nat_reader
        summary["k_ref_natural_reader"] = summary.get("k_ref_natural_reader", {})
        summary["k_ref_natural_reader"][tier] = nat_reader
    # amendment A: a zero natural count (a discrete cut at its ceiling) falls back to 7/80
    summary["k_ref_effective"] = {}
    for tier in TIERS:
        summary["k_ref_effective"][tier] = 7 if summary["k_ref"][tier] == 0 else summary["k_ref"][tier]

    # matched flags (k_ref per tier) on draw 1; the reader-start/oracle arms are also cut on draw 2
    matched = {}
    curve = {}
    for tier in TIERS:
        k = summary["k_ref_effective"][tier]
        for arm in ARMS:
            matched[(tier, arm)], thr = cut_flags(d1, tier, arm, k)
            summary["thresholds"].setdefault(tier, {})[arm] = thr
        for kk in (5, 7, 10):
            for arm in ARMS:
                curve[(tier, arm, kk)] = cut_flags(d1, tier, arm, kk)[0]
    for tier in TIERS:
        for arm in ("dpvo_reader_start", "dpvo_oracle"):
            matched[(tier, arm, 2)], _ = cut_flags(d2, tier, arm, summary["k_ref_effective"][tier])

    def cell(sub, tier, arm, mode="matched"):
        if mode == "matched":
            fl = matched[(tier, arm)]
        elif mode == "natural":
            fl = nat[(tier, arm)]
        else:
            fl = curve[(tier, arm, int(mode))]
        return rate(sub, fl)

    strata = {
        "all": lambda r: True,
        "forest": lambda r: r["family"] == "forest_ridge",
        "urban": lambda r: r["family"] == "urban_checker",
        "industrial": lambda r: r["family"] == "industrial",
        "mount_0": lambda r: r["mount"] == 0.0,
        "mount_15": lambda r: r["mount"] == 15.0,
        "mount_30": lambda r: r["mount"] == 30.0,
        "mount_45": lambda r: r["mount"] == 45.0,
        "high_mount": lambda r: r["mount"] in (30.0, 45.0),
    }
    census = []
    for mode in ("matched", "natural", "5", "7", "10"):
        for tier in TIERS:
            for sname, fn in strata.items():
                for cond in conds:
                    sub = [r for r in d1 if fn(r) and r["condition"] == cond]
                    if not sub:
                        continue
                    row = {"mode": mode, "tier": tier, "stratum": sname, "condition": cond,
                           "n": len(sub)}
                    for arm in ARMS:
                        row[arm] = cell(sub, tier, arm, mode)
                    census.append(row)
    with open(OUT / "census.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(census[0].keys()))
        w.writeheader(); w.writerows(census)

    # headline tables: all / forest / high_mount, matched
    table = {}
    for sname in ("all", "forest", "high_mount"):
        fn = strata[sname]
        table[sname] = {}
        for tier in TIERS:
            table[sname][tier] = {}
            for cond in conds:
                sub = [r for r in d1 if fn(r) and r["condition"] == cond]
                if not sub:
                    continue
                table[sname][tier][cond] = {arm: cell(sub, tier, arm) for arm in ARMS}
    summary["table_matched"] = table

    # by-mount matched, removals and feasible only (the two strata questions)
    bymount = {}
    for tier in TIERS:
        bymount[tier] = {}
        for m in mounts:
            sub = [r for r in d1 if r["mount"] == m]
            bymount[tier][str(m)] = {
                "feasible": {arm: cell([r for r in sub if r["condition"] == "feasible"],
                                       tier, arm) for arm in ARMS},
                "tilt_removed": {arm: cell([r for r in sub if r["condition"] == "tilt_removed"],
                                           tier, arm) for arm in ARMS},
                "mirrored_roll": {arm: cell([r for r in sub if r["condition"] == "mirrored_roll"],
                                            tier, arm) for arm in ARMS},
            }
    summary["by_mount_matched"] = bymount

    summary["abstentions"] = {
        cond: {f"{t}|{a}": int(sum(not np.isfinite(r[f"{t}|{a}"]) for r in d1
                                   if r["condition"] == cond))
               for t in TIERS for a in ARMS}
        for cond in conds}
    # medians of every statistic by condition, arm
    summary["medians"] = {
        cond: {f"{t}|{a}": float(np.nanmedian([r[f"{t}|{a}"] for r in d1
                                               if r["condition"] == cond]))
               for t in TIERS for a in ARMS}
        for cond in conds}

    # draw-2 matched for the two robustness arms
    summary["draw2_matched"] = {
        tier: {cond: {arm: cell([r for r in d2 if r["condition"] == cond], tier, arm)
                      for arm in ("dpvo_reader_start", "dpvo_oracle")}
               for cond in conds}
        for tier in TIERS}

    # start-error sensitivity: the 2-degree arm against the oracle, by stratum
    sens = {}
    for sname, fn in strata.items():
        sens[sname] = {}
        for tier in TIERS:
            sub = [r for r in d1 if fn(r)]
            sens[sname][tier] = {
                "median_oracle": float(np.nanmedian([r[f"{tier}|dpvo_oracle"] for r in sub])),
                "median_oracle_2deg": float(np.nanmedian([r[f"{tier}|dpvo_oracle_2deg"]
                                                          for r in sub])),
            }
    summary["start_sensitivity"] = sens

    # dynamic-clip census (Part D input; descriptive here)
    summary["dynamic"] = {
        "n_all": int(sum(r["dynamic"] for r in d1)),
        "n_by_condition": {c: int(sum(r["dynamic"] and r["condition"] == c for r in d1))
                           for c in conds},
        "n_by_family": {f: int(sum(r["dynamic"] and r["family"] == f for r in d1))
                        for f in families},
    }
    summary["note"] = (
        "matched = every arm cut at the (k_ref+1)-th largest pooled feasible score, k_ref = the "
        "reader arm's natural leave-one-seed-out feasible false-reject count for that tier; "
        "natural = leave-one-seed-out 95th percentile per clip; curve = fixed pooled feasible "
        "false-reject counts 5/7/10 of 80. Draw 1 unless 'draw2_matched'. Reader, level, "
        "reader-start, oracle, oracle+2deg(median of 8).")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    print(json.dumps({"k_ref": summary["k_ref"]}, indent=1))
    for sname in ("all", "forest", "high_mount"):
        print("\n==", sname)
        for cond in conds:
            if cond in table[sname]["F0"]:
                print(cond, table[sname]["F0"][cond])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
