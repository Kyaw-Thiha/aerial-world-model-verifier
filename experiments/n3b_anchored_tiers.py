"""Track N3, Part B: verdict tiers on the lean propagated from a path anchor (simulated set).

Pre-registration `configs/n3b_anchored_tiers_prereg.yaml`, committed before any anchored score
existed. Thresholds, matching rule, arms and success criteria are fixed there.

Every number is a CPU reanalysis of the stored Track N caches. The required side, the reader
scores, the scored frame grid and the DPVO poses are the same arrays N05/N07 used; the F0/C2/R
statistics are the same functions. As a guard, the reader/oracle/reader-start F0 values are
compared against `results/n2/n07/per_clip.csv` and the run fails if they disagree.

Run:  PYTHONPATH=. pixi run python experiments/n3b_anchored_tiers.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.perception.frames import camera_to_body_pitched
from src.verdict.anchor import (E3, angle_between_deg, find_windows,
                                first_sustained_motion, implied_thrust_axis,
                                implied_tilt_deg, last_sustained_motion,
                                propagate_from, propagate_from_nearest, select_anchor,
                                snap_to_grid, up_cam_at_anchor)
from src.depicted.clips import up_body_from_lean
from src.physics.flatness import angle_between_deg as axis_angle_deg
from src.verdict.lean import depicted_lean_from_gravity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n3/n3b"
sys_path = str(ROOT / "experiments")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from n07_sim_tiers import tier_scores as TIER_SCORES  # noqa: E402  (N05's C2/R, unchanged)
GT = ROOT / "data/n05/gt"
SCORES = ROOT / "data/n05/scores"
DRAW1 = ROOT / "data/n05/poses/sim200/scale1"
DRAW2 = ROOT / "data/n05/poses/sim200/scale1_rep2"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"
N07 = ROOT / "results/n2/n07/per_clip.csv"
WINDOWS = ROOT / "results/n3/n3a/windows.csv"

G0 = 9.80665
SIGMA_N_DEG = 1.910587818373582
EPS_SIGN = float(np.sin(np.deg2rad(SIGMA_N_DEG)))
WIN, HOP, MIN_USABLE, MIN_SIGN = 8, 4, 6, 2
TIERS_ANCHOR = {"P1": (2.0, 1.0), "P2": (4.0, 1.0), "P3": (6.0, 1.0)}
TIERS = ("F0", "C2", "R")
ARMS = ("reader", "level", "reader_start", "oracle", "oracle_2deg",
        "anchor1", "anchor1_P2", "anchor1_P3", "anchor_multi_P3", "oracle_window")
CONDS = ("feasible", "tilt_removed", "mirrored_roll", "tilt_scaled", "partial", "thrust_limit")
FAMILIES = ("urban_checker", "forest_ridge", "industrial")
MOUNTS = (0.0, 15.0, 30.0, 45.0)
K_REF = 8


# ------------------------------------------------------------------ tier statistics (N07 identical)
def f0_stat(l_dep, l_req, ok):
    """K03 fixed-calibration thrust-axis angle: clip median, no nuisance (N07 identical)."""
    if ok.sum() < 8:
        return float("nan")
    a = up_body_from_lean(np.asarray(l_dep, float)[ok])
    b = up_body_from_lean(np.asarray(l_req, float)[ok])
    good = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if good.sum() < 8:
        return float("nan")
    return float(np.median(axis_angle_deg(a[good], b[good])))


def _windows(n):
    if n <= WIN:
        return [(0, n)]
    return [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]


def _sign_agreement(a, b, eps):
    keep = (np.abs(a) > eps) & (np.abs(b) > eps)
    if keep.sum() == 0:
        return 0.5, 0
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep]))), int(keep.sum())


def c2_stat(l_dep, l_req, vreg, ok):
    """N05/N07 C2 and R, reused byte-identically from the N07 module."""
    c2, r, _ = TIER_SCORES(l_dep, l_req, vreg, ok)
    return c2, r


def all_tiers(l_dep, l_req, vreg, ok):
    c2, r = c2_stat(l_dep, l_req, vreg, ok)
    return {"F0": f0_stat(l_dep, l_req, ok), "C2": c2, "R": r}


# ------------------------------------------------------------------ per clip
def true_c2w(gt, R_cb, t):
    idx = np.clip(np.round(np.asarray(t) * float(np.asarray(gt["gt_hz"]).ravel()[0])).astype(int),
                  0, len(np.asarray(gt["R_bw_cam"])) - 1)
    return np.einsum("nij,jk->nik", np.asarray(gt["R_bw_cam"], float)[idx], R_cb)


def anchors_for(gt, R_cb, fidx, tiers=TIERS_ANCHOR):
    """Re-derive the N3A anchor frames for this clip (P1/P2/P3), plus every P3 window frame."""
    t = np.asarray(gt["t"], float)
    theta = implied_tilt_deg(np.asarray(gt["acc"], float), np.asarray(gt["vel"], float))
    speed = np.linalg.norm(np.asarray(gt["vel"], float), axis=1)
    t_first = first_sustained_motion(t, speed)
    t_last = last_sustained_motion(t, speed)
    hover_only = not np.isfinite(t_first)
    inside = (t >= fidx[0] / 30.0 - 1e-9) & (t <= fidx[-1] / 30.0 + 1e-9)
    out = {}
    for tier, (thr, dur) in tiers.items():
        wins = find_windows(theta, t, thr, dur, valid=inside)
        wins = wins if hover_only else [w for w in wins if t[w.start] >= t_first
                                        and t[w.stop - 1] <= t_last]
        best = select_anchor(wins)
        out[tier] = {"t_a": float(t[best.argmin]) if best else None,
                     "n_windows": len(wins),
                     "all_t": [float(t[w.argmin]) for w in wins]}
    return out


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

    d = np.load(draw_dir / f"{cid}.npz")
    R = np.asarray(d["c2w"], float)[:, :3, :3]
    t_pose = np.asarray(d["t"], float)
    l_reader = np.asarray(sc["lean_est"], float)
    up0_reader = np.asarray(sc["up_est"], float)[0]
    up0_reader = up0_reader / np.linalg.norm(up0_reader)
    Rtrue = true_c2w(gt, R_cb, t)
    up0_true = Rtrue[0].T @ E3
    u_anchor = up_cam_at_anchor(R_cb)

    ups = {
        "level": propagate_from(R, 0, R_cb.T @ E3, fidx),
        "reader_start": propagate_from(R, 0, up0_reader, fidx),
        "oracle": propagate_from(R, 0, up0_true, fidx),
    }
    anc = anchors_for(gt, R_cb, fidx)
    extra = {}
    for tag, tier in (("anchor1", "P1"), ("anchor1_P2", "P2"), ("anchor1_P3", "P3")):
        a = anc[tier]["t_a"]
        if a is None:
            ups[tag] = np.full((len(fidx), 3), np.nan)
        else:
            ia = snap_to_grid(a, t_pose)
            ups[tag] = propagate_from(R, ia, u_anchor, fidx)
            extra[f"lever|{tag}"] = float(np.median(np.abs(t - a)))
            extra[f"anchor_t|{tag}"] = float(a)
            # the TRUE up at the same frame: the anchor's angular error, isolated from the drift
            if tag == "anchor1_P3":
                utrue = Rtrue[int(np.argmin(np.abs(t - a)))].T @ E3
                ups["oracle_window"] = propagate_from(R, ia, utrue, fidx)
    if anc["P3"]["n_windows"]:
        idxs = [snap_to_grid(x, t_pose) for x in anc["P3"]["all_t"]]
        ups["anchor_multi_P3"] = propagate_from_nearest(R, idxs, u_anchor, fidx)
        extra["n_anchor_windows_P3"] = anc["P3"]["n_windows"]
    else:
        ups["anchor_multi_P3"] = np.full((len(fidx), 3), np.nan)

    leans = {"reader": l_reader}
    for k, u in ups.items():
        leans[k] = depicted_lean_from_gravity(u, R_cb)
    for arm in ARMS:
        leans.setdefault(arm, np.full((len(fidx), 3), np.nan))

    per_draw = {"F0": [], "C2": [], "R": []}
    if sens:
        tang, bino = tangent_frame(up0_true)
        th = np.deg2rad(2.0)
        for _ in range(8):
            phi = rng.uniform(0.0, 2.0 * np.pi)
            axis = np.cos(phi) * tang + np.sin(phi) * bino
            u0 = np.cos(th) * up0_true + np.sin(th) * axis
            u0 = u0 / np.linalg.norm(u0)
            l = depicted_lean_from_gravity(propagate_from(R, 0, u0, fidx), R_cb)
            ok2 = np.isfinite(l).all(1) & np.isfinite(lr).all(1) & np.isfinite(vreg).all(1)
            st = all_tiers(l, lr, vreg, ok2)
            for tier in TIERS:
                per_draw[tier].append(st[tier])

    out = {"clip_id": cid, "family": row["family"], "scene_seed": row["scene_seed"],
           "mount": float(row["mount"]), "condition": row["condition"],
           "factor": row["factor"], "interval_fraction": row["interval_fraction"],
           "dynamic": int(float(row["peak_tilt_req_deg"]) >= 15.0),
           "peak_tilt_req_deg": float(row["peak_tilt_req_deg"]),
           **{k: v for k, v in extra.items()}}
    for arm in ARMS:
        l = leans[arm]
        ok = np.isfinite(l).all(1) & np.isfinite(lr).all(1) & np.isfinite(vreg).all(1)
        st = all_tiers(l, lr, vreg, ok)
        for tier in TIERS:
            out[f"{tier}|{arm}"] = st[tier]
        out[f"n|{arm}"] = int(ok.sum())
    for tier in TIERS:
        v = np.asarray(per_draw[tier], float)
        out[f"{tier}|oracle_2deg"] = (float(np.nanmedian(v)) if v.size and np.isfinite(v).any()
                                      else float("nan"))
    return out


def tangent_frame(u):
    t = np.cross(u, E3)
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(u, np.array([1.0, 0.0, 0.0]))
    t = t / np.linalg.norm(t)
    return t, np.cross(u, t)


# ------------------------------------------------------------------ thresholds and rates
def cut_flags(rows, tier, arm, k):
    feas = [r for r in rows if r["condition"] == "feasible"]
    vals = sorted([f[f"{tier}|{arm}"] for f in feas if np.isfinite(f[f"{tier}|{arm}"])],
                  reverse=True)
    if not vals:
        return {}, float("nan"), 0
    kk = min(k, len(vals))
    thr = vals[kk] if kk < len(vals) else vals[-1]
    flags = {r["clip_id"]: bool(np.isfinite(r[f"{tier}|{arm}"])
                                and r[f"{tier}|{arm}"] > thr) for r in rows}
    return flags, float(thr), len(vals)


def rate(sub, flags):
    v = [flags[r["clip_id"]] for r in sub]
    return f"{int(sum(v))}/{len(v)}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "results/n05/clips.csv")))
    assert len(rows) == 200 and all(r["clip_id"].startswith("i02_dev") for r in rows)
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "dev"}
    for r in rows:
        r["peak_tilt_req_deg"] = man[r["clip_id"]]["peak_tilt_req_deg"]

    rng = np.random.default_rng(1)
    d1 = [build(r, DRAW1, rng) for r in rows]
    keys = list(d1[0].keys())
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(d1)

    # draw 2 for the anchored arms only (robustness), reported in the summary
    d2 = []
    for r in rows:
        rec = build(r, DRAW2, rng, sens=False)
        d2.append(rec)

    with open(OUT / "per_clip_draw2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(d2)

    # ------------------------------------------------------------------ cross-check vs N07
    ref = {r["clip_id"]: r for r in csv.DictReader(open(N07)) if r["draw"] == "1"}
    checks = {}
    n07_names = {"reader": "reader", "level": "dpvo_level", "reader_start": "dpvo_reader_start",
                 "oracle": "dpvo_oracle", "oracle_2deg": "dpvo_oracle_2deg"}
    for arm, n07 in n07_names.items():
        diffs = [abs(float(a[f"F0|{arm}"]) - float(ref[a["clip_id"]][f"F0|{n07}"]))
                 for a in d1 if np.isfinite(float(ref[a["clip_id"]][f"F0|{n07}"]))
                 and np.isfinite(a[f"F0|{arm}"])]
        checks[f"F0|{arm}"] = float(np.max(diffs)) if diffs else float("nan")
    assert max(checks.values()) < 1e-6, f"N07 cross-check failed: {checks}"

    # the anchor derivation must equal N3A's table
    win = {(r["unit"], r["tier"]): r for r in csv.DictReader(open(WINDOWS))
           if r["kind"] == "sim"}
    anc_checks = []
    for a in d1:
        for tag, tier in (("anchor1", "P1"), ("anchor1_P2", "P2"), ("anchor1_P3", "P3")):
            got = a.get(f"anchor_t|{tag}")
            exp = win[(a["clip_id"], tier)]["anchor_t_s"]
            if (got is None) != (exp == ""):
                anc_checks.append((a["clip_id"], tier, got, exp))
                continue
            if got is not None and abs(got - float(exp)) > 1e-9:
                anc_checks.append((a["clip_id"], tier, got, exp))
    assert not anc_checks, f"anchor table mismatch: {anc_checks[:3]}"

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
        "fore_mount_high": lambda r: r["family"] == "forest_ridge"
        and r["mount"] in (30.0, 45.0),
    }

    summary = {"config": {"k_ref": K_REF, "tiers_anchor": TIERS_ANCHOR,
                          "arms": list(ARMS), "cross_check_n07": checks,
                          "cross_check_anchor_table": "pass"},
               "coverage": {}, "matched": {}, "curve": {}, "medians": {}, "tail": {},
               "abstentions": {}, "draw2": {}, "subset": {}}

    # coverage of each anchored arm
    for arm in ARMS:
        flags = {"has": {r["clip_id"]: bool(np.isfinite(r[f"F0|{arm}"])) for r in d1}}
        summary["coverage"][arm] = {
            s: rate([r for r in d1 if fn(r)], flags["has"]) for s, fn in strata.items()}
        summary["coverage"][arm]["by_condition"] = {
            c: rate([r for r in d1 if r["condition"] == c], flags["has"]) for c in CONDS}

    # matched point: k = 8 pooled feasible false-rejects for every arm/tier
    matched = {}
    for tier in TIERS:
        for arm in ARMS:
            matched[(tier, arm)], thr, nf = cut_flags(d1, tier, arm, K_REF)
            summary["matched"].setdefault(tier, {})[arm] = {
                "threshold": thr, "n_feasible_scored": nf,
                "feasible_fr": rate([r for r in d1 if r["condition"] == "feasible"],
                                    matched[(tier, arm)]),
                **{c: rate([r for r in d1 if r["condition"] == c], matched[(tier, arm)])
                   for c in CONDS if c != "feasible"}}
    # per stratum detections at the matched point
    summary["matched_strata"] = {}
    for tier in TIERS:
        summary["matched_strata"][tier] = {}
        for s, fn in strata.items():
            sub = [r for r in d1 if fn(r)]
            summary["matched_strata"][tier][s] = {
                arm: {c: rate([r for r in sub if r["condition"] == c], matched[(tier, arm)])
                      for c in CONDS}
                for arm in ARMS}

    # curve: fixed pooled feasible false-reject counts 5 / 7 / 10
    for k in (5, 7, 10):
        for tier in ("F0",):
            for arm in ARMS:
                fl, thr, nf = cut_flags(d1, tier, arm, k)
                summary["curve"][f"{tier}|{arm}|k{k}"] = {
                    "threshold": thr, "n_feasible_scored": nf,
                    **{c: rate([r for r in d1 if r["condition"] == c], fl) for c in CONDS},
                    "forest": {c: rate([r for r in d1 if r["family"] == "forest_ridge"
                                        and r["condition"] == c], fl) for c in CONDS},
                    "high_mount": {c: rate([r for r in d1 if r["mount"] in (30.0, 45.0)
                                            and r["condition"] == c], fl) for c in CONDS}}
    # medians of F0 by condition and stratum, and the >20 deg tail
    for arm in ARMS:
        summary["medians"][arm] = {
            c: float(np.nanmedian([r[f"F0|{arm}"] for r in d1 if r["condition"] == c]))
            for c in CONDS}
        summary["medians"][arm]["by_mount_feasible"] = {
            str(m): float(np.nanmedian([r[f"F0|{arm}"] for r in d1
                                        if r["condition"] == "feasible" and r["mount"] == m]))
            for m in MOUNTS}
        summary["tail"][arm] = {
            "feasible_f0_gt_20": int(sum(1 for r in d1 if r["condition"] == "feasible"
                                         and np.isfinite(r[f"F0|{arm}"])
                                         and r[f"F0|{arm}"] > 20.0)),
            "feasible_f0_gt_15": int(sum(1 for r in d1 if r["condition"] == "feasible"
                                         and np.isfinite(r[f"F0|{arm}"])
                                         and r[f"F0|{arm}"] > 15.0)),
            "by_mount_gt_20": {str(m): int(sum(1 for r in d1 if r["condition"] == "feasible"
                                              and r["mount"] == m
                                              and np.isfinite(r[f"F0|{arm}"])
                                              and r[f"F0|{arm}"] > 20.0)) for m in MOUNTS}}

    # the mechanism table: on clips that HAVE an anchor, how good is the anchored lean?
    for tag in ("anchor1", "anchor1_P2", "anchor1_P3", "anchor_multi_P3", "oracle_window"):
        sub = [r for r in d1 if np.isfinite(r[f"F0|{tag}"])]
        summary["subset"][tag] = {
            "n_clips": len(sub),
            "median_F0_by_condition": {
                c: (float(np.nanmedian([r[f"F0|{tag}"] for r in sub if r["condition"] == c]))
                    if any(r["condition"] == c for r in sub) else None) for c in CONDS},
            "n_by_condition": {c: int(sum(r["condition"] == c for r in sub)) for c in CONDS},
            "median_F0_by_mount": {
                str(m): (float(np.nanmedian([r[f"F0|{tag}"] for r in sub if r["mount"] == m]))
                         if any(r["mount"] == m for r in sub) else None) for m in MOUNTS},
        }
    # paired comparison on the anchor subset: anchor vs reader_start / oracle / reader, same clips
    paired = {}
    for tag in ("anchor1_P2", "anchor1_P3", "anchor_multi_P3", "oracle_window"):
        sub = [r for r in d1 if r["condition"] == "feasible" and np.isfinite(r[f"F0|{tag}"])]
        paired[tag] = {other: {
            "n": len(sub),
            "median_delta_deg": float(np.nanmedian([r[f"F0|{tag}"] - r[f"F0|{other}"]
                                                    for r in sub])),
            "wins": int(sum(r[f"F0|{tag}"] < r[f"F0|{other}"] for r in sub)),
            "by_mount": {str(m): float(np.nanmedian(
                [r[f"F0|{tag}"] - r[f"F0|{other}"] for r in sub if r["mount"] == m]))
                if any(r["mount"] == m for r in sub) else None for m in MOUNTS}}
            for other in ("reader", "reader_start", "oracle", "level")}
    summary["paired_feasible"] = paired

    # descriptive within-subset operating point: the cut at the worst ANCHORED feasible residual
    # (a fair reading of the mechanism, since the pooled matched point is degenerate for an arm
    # that abstains on 88-96 % of the feasible clips; labelled descriptive everywhere)
    subset_op = {}
    for tag in ("anchor1", "anchor1_P2", "anchor1_P3", "anchor_multi_P3", "oracle_window"):
        sub = [r for r in d1 if np.isfinite(r[f"F0|{tag}"])]
        feas = [r for r in sub if r["condition"] == "feasible"]
        if not feas:
            continue
        cut = float(max(r[f"F0|{tag}"] for r in feas))
        subset_op[tag] = {
            "cut_deg": cut, "n_feasible": len(feas),
            "n_flagged_feasible": int(sum(r[f"F0|{tag}"] > cut for r in feas)),
            **{c: f"{sum(1 for r in sub if r['condition'] == c and r[f'F0|{tag}'] > cut)}"
                  f"/{sum(1 for r in sub if r['condition'] == c)}" for c in CONDS},
            "note": ("cut = the worst anchored feasible residual, so 0 false-rejects by "
                     "construction on the anchored subset; every defect rate is over the clips "
                     "with an anchor, not over the full condition")}
    summary["anchored_subset_operating_point"] = subset_op

    # draw-2 robustness for the anchored arms
    for tag in ("anchor1_P2", "anchor1_P3", "anchor_multi_P3"):
        sub = [r for r in d2 if np.isfinite(r[f"F0|{tag}"])]
        summary["draw2"][tag] = {
            "n_clips": len(sub),
            "median_F0_feasible": (float(np.nanmedian([r[f"F0|{tag}"] for r in sub
                                                       if r["condition"] == "feasible"]))
                                   if any(r["condition"] == "feasible" for r in sub) else None),
            "feasible_f0_gt_20": int(sum(1 for r in sub if r["condition"] == "feasible"
                                         and r[f"F0|{tag}"] > 20.0))}

    # success criteria, scored mechanically
    def cell(tier, arm, cond, stratum="all"):
        return rate([r for r in d1 if strata[stratum](r) and r["condition"] == cond],
                    matched[(tier, arm)])

    def cnt(tier, arm, cond, stratum="all"):
        s = cell(tier, arm, cond, stratum)
        return int(s.split("/")[0])

    crit = {}
    for tag, tier_name in (("anchor1", "P1"), ("anchor1_P2", "P2"), ("anchor1_P3", "P3")):
        b2a = cnt("F0", tag, "tilt_removed", "forest")
        b2b = cnt("F0", tag, "feasible", "forest")
        b3a = cnt("F0", tag, "tilt_removed", "all")
        b3b = cnt("F0", tag, "mirrored_roll", "all")
        b5 = cnt("F0", tag, "tilt_removed", "forest") - cnt("F0", "reader_start",
                                                           "tilt_removed", "forest")
        b5b = cnt("F0", "reader_start", "feasible", "forest") - cnt("F0", tag, "feasible",
                                                                   "forest")
        crit[tag] = {
            "tier_config": tier_name,
            "B2_forest": {"removals": f"{b2a}/9", "false_rejects": f"{b2b}/26",
                          "pass": bool(b2a >= 8 and b2b <= 4)},
            "B3_overall": {"removals": f"{b3a}/27", "mirrors": f"{b3b}/27",
                           "pass": bool(b3a >= 18 and b3b >= 15)},
            "B4_drift": {"feasible_gt_20": summary["tail"][tag]["feasible_f0_gt_20"],
                         "cut_deg": summary["matched"]["F0"][tag]["threshold"],
                         "pass": bool(summary["tail"][tag]["feasible_f0_gt_20"] <= 3
                                      and summary["matched"]["F0"][tag]["threshold"] <= 12.0)},
            "B5_vs_reader_start": {"delta_forest_removals": b5,
                                   "delta_forest_false_rejects": b5b,
                                   "pass": bool(b5 >= 3 or b5b >= 2)},
        }
    summary["criteria"] = crit
    summary["criteria"]["B1_gate"] = {
        "n3a_availability_P1": "8/200 clips, 0/131 dynamic, 2/26 forest feasible -> FAIL",
        "consequence": "Part B is DESCRIPTIVE (pre-registered): every detection table carries its "
                       "coverage, and no B2-B5 pass is a confirmatory success.",
    }
    summary["note"] = (
        "matched = every arm cut at its own (k+1)-th largest pooled feasible score, k=8, so each "
        "arm flags 8 feasible clips where it has 8 scoreable ones; abstentions count as "
        "non-detections. `anchor1*` abstain on clips with no anchor at that tier. "
        "`oracle_window` uses the true up at the P3 anchor frame of the same clip (exploratory).")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    print(json.dumps({"n07_checks": checks,
                      "coverage": {a: summary["coverage"][a]["all"] for a in ARMS},
                      "median_F0_feasible": {a: summary["medians"][a]["feasible"] for a in ARMS},
                      "tail20": {a: summary["tail"][a]["feasible_f0_gt_20"] for a in ARMS},
                      "cut": {a: summary["matched"]["F0"][a]["threshold"] for a in ARMS}},
                     indent=1))
    for s in ("all", "forest", "high_mount"):
        print("==", s)
        for arm in ("reader", "reader_start", "anchor1_P3", "anchor1_P2", "anchor_multi_P3",
                    "oracle_window", "oracle"):
            print(" ", arm, {c: summary["matched_strata"]["F0"][s][arm][c]
                             for c in ("feasible", "tilt_removed", "mirrored_roll")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
