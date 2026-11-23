"""G07 - the frozen held-out evaluation, run once.

Recipe: ``configs/g06_eval_frozen.yaml`` (frozen after the dev pass, before any held-out score
existed).  Pre-registration and success criteria: ``configs/g06_segments_prereg.yaml``.

Fits every threshold on the FIT population (simulated: all feasible dev clips; real: all
gate-kept dev clips) and applies it unchanged to the held-out clips. Evaluates exactly the arms
the frozen recipe lists, the whole-clip median baseline included, and evaluates the pre-registered
success criteria once with the denominators the held-out set actually provides.

Nothing here chooses a knob: arms, tiers, lengths, hops, persistence, alpha and q come from the
frozen recipe; the window code is round 1/2's, unchanged.

    PYTHONPATH=. python experiments/g07_heldout.py \
        --scores-dev results/g06/scores --scores-heldout results/g07/scores
    PYTHONPATH=. python experiments/g07_heldout.py --smoke-dev      # code check on DEV clips
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from experiments.g3_data import load_split, read_manifest
from experiments.g3_extra import bh_reject, empirical_p, refine_region
from src.verdict.fit import _shift_series
from src.verdict.segments import (clip_max, clip_max_persistent, clip_median_whole,
                                  flagged_windows, flagged_windows_persistent,
                                  frame_residual_deg, interval_scores, merge_intervals,
                                  window_statistics)

FROZEN = Path("configs/g06_eval_frozen.yaml")
CLIP_ROOT = Path("data/processed/track_d/clips")
GATE_CFG = Path("configs/gate_frozen.yaml")
ALPHA = 0.05
SEED = 1
N_BOOT = 5000
FPS_SIM = 30.0
LENGTHS = (8, 12, 16)
REAL_FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
REAL_CONTROLS = ("yaw_control", "null_edit")
REAL_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
DEV_REAL_FLIGHTS = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
                    "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]


# ------------------------------------------------------------------ frozen recipe
def kind_of(a: dict) -> str:
    n = a["name"]
    if n.startswith("a2_bh"):
        return "bh"
    if n == "a5_joint":
        return "joint"
    if n == "a4_lag_coh12":
        return "lag"
    if n == "a3_refined":
        return "refined"
    if n == "median_baseline":
        return "median"
    return "window"


# The frozen recipe's arm list, read by name.  A name fixes the arm's knobs because the frozen
# file states them redundantly and its flow mappings are not valid YAML; the assertions below
# check the name against the line's own text, so a typo cannot pass silently.
ARM_TABLE = {
    "coh12_p1": dict(tier="coherence", length=12, persistence=1, kind="window"),
    "coh12_p2": dict(tier="coherence", length=12, persistence=2, kind="window"),
    "coh8_p1": dict(tier="coherence", length=8, persistence=1, kind="window"),
    "res12_p1": dict(tier="residual", length=12, persistence=1, kind="window"),
    "res8_p1": dict(tier="residual", length=8, persistence=1, kind="window"),
    "res8_p2": dict(tier="residual", length=8, persistence=2, kind="window"),
    "sign12_p1": dict(tier="sign", length=12, persistence=1, kind="window"),
    "median_baseline": dict(tier="median", length=0, persistence=1, kind="median"),
    "a2_bh_coh12_q10": dict(tier="coherence", length=12, persistence=1, kind="bh", q=0.10),
    "a2_bh_res8_q10": dict(tier="residual", length=8, persistence=1, kind="bh", q=0.10),
    "a5_joint": dict(kind="joint"),
    "a4_lag_coh12": dict(tier="coherence", length=12, persistence=1, kind="lag"),
    "a3_refined": dict(kind="refined"),
}


def _arm_from_line(line: str) -> dict:
    import re

    m = re.search(r"name:\s*([^,\s}]+)", line)
    if not m:
        raise SystemExit(f"cannot read an arm name from: {line!r}")
    name = m.group(1)
    if name not in ARM_TABLE:
        raise SystemExit(f"frozen arm {name!r} has no entry in ARM_TABLE")
    a = {"name": name, **ARM_TABLE[name], "hop": 4}
    # cross-check only the fields the frozen line states; the a4/a5/a3 arms are described by
    # rule text, and their knobs come from the pre-registration ("primary" = coh12_p1).
    if a.get("length") and "length:" in line and f"length: {a['length']}" not in line:
        raise SystemExit(f"arm {name}: length {a['length']} not found in the frozen line")
    if a.get("q") is not None and "q:" in line and f"q: {a['q']:g}" not in line:
        raise SystemExit(f"arm {name}: q {a['q']} not found in the frozen line")
    if a["kind"] == "window" and "persistence:" in line \
            and f"persistence: {a['persistence']}" not in line:
        raise SystemExit(f"arm {name}: persistence not found in the frozen line")
    return a


def load_cfg() -> dict:
    """Read the frozen recipe.

    The ``arms_evaluated`` block nests braces inside flow mappings (``lengths {8,12,16}``),
    which is not valid YAML; that block is therefore parsed line by line and the rest of the
    file with the block removed.  No value is changed and the frozen file is not edited; the
    parse mode is recorded in results/g07 and in the report.
    """
    lines = FROZEN.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("arms_evaluated:"))
    end = next(i for i in range(start + 1, len(lines))
               if lines[i] and not lines[i].startswith(" "))
    arm_lines = [l for l in lines[start + 1:end] if l.strip().startswith("- {")]
    rest = lines[:start] + lines[end:]
    cfg = yaml.safe_load("\n".join(rest))
    cfg["arms_evaluated"] = [_arm_from_line(l) for l in arm_lines]
    cfg["_parse_mode"] = "arms_evaluated parsed line by line (flow mappings are not valid YAML)"
    return cfg


def load_arms() -> list[dict]:
    cfg = load_cfg()
    arms = []
    for a in cfg["arms_evaluated"]:
        a = dict(a)
        a.setdefault("persistence", 1)
        a.setdefault("hop", 4)
        a["kind"] = kind_of(a)
        arms.append(a)
    return arms


# ------------------------------------------------------------------ clip duck type
class C:
    """The frozen window code's duck type: t, lean_dep, lean_req, tilt_req_deg, truth, yaw."""

    def __init__(self, lean_dep, lean_req, tilt_req, t, truth, yaw=None, clip_id=""):
        self.lean_dep = np.asarray(lean_dep, float)
        self.lean_req = np.asarray(lean_req, float)
        self.tilt_req_deg = np.asarray(tilt_req, float)
        self.t = np.asarray(t, float)
        self.truth = np.asarray(truth, bool)
        self._yaw = np.zeros(len(self.t)) if yaw is None else np.asarray(yaw, float)
        self.clip_id = clip_id

    def __len__(self):
        return len(self.t)

    @property
    def depicted_lean(self):
        return self.lean_dep

    @property
    def yaw(self):
        return self._yaw


# ------------------------------------------------------------------ statistics
def window_values(stats, tier) -> np.ndarray:
    out = [w.get(tier) for w in stats if not w.abstained and np.isfinite(w.get(tier))]
    return np.asarray(out, float)


def arm_value(stats, a: dict) -> float:
    if a["persistence"] > 1:
        return clip_max_persistent(stats, a["tier"], a["persistence"])
    return clip_max(stats, a["tier"])


def arm_mask(stats, a: dict, tau: float) -> np.ndarray:
    if a["persistence"] > 1:
        return flagged_windows_persistent(stats, a["tier"], tau, a["persistence"])
    return flagged_windows(stats, a["tier"], tau)


def quantile(vals, alpha: float = ALPHA) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return float(np.quantile(v, 1.0 - alpha)) if len(v) >= 8 else float("nan")


def robust(vals) -> tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 8:
        return float("nan"), float("nan")
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(np.median(v)), float(max((hi - lo) / 3.92, 1e-6))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - s) / d), float((c + s) / d)


def clustered_rate(flags_by_cluster: dict) -> dict:
    vals = {k: np.asarray(v, float) for k, v in flags_by_cluster.items() if v}
    allv = np.concatenate(list(vals.values())) if vals else np.array([])
    k, n = int(np.nansum(allv)), int(len(allv))
    out = {"k": k, "n": n, "rate": float(k / n) if n else float("nan"),
           "wilson": wilson(k, n), "n_clusters": len(vals)}
    if len(vals) < 2:
        out["boot"] = (float("nan"), float("nan"))
        out["clusters_few_warning"] = True
        return out
    rng = np.random.default_rng(SEED)
    ks = list(vals)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(ks), size=len(ks), replace=True)
        draws.append(float(np.mean(np.concatenate([vals[ks[i]] for i in pick]))))
    out["boot"] = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    out["clusters_few_warning"] = len(vals) < 10
    return out


def best_lag(clip) -> float:
    dep = np.linalg.norm(np.asarray(clip.lean_dep, float), axis=1)
    best, lag_best = -2.0, 0.0
    for lag in np.arange(-0.150, 0.1501, 0.025):
        req = np.linalg.norm(_shift_series(np.asarray(clip.lean_req, float), clip.t, float(lag)),
                             axis=1)
        ok = np.isfinite(dep) & np.isfinite(req)
        if ok.sum() < 4 or np.std(dep[ok]) < 1e-9 or np.std(req[ok]) < 1e-9:
            continue
        c = float(np.corrcoef(dep[ok], req[ok])[0, 1])
        if c > best:
            best, lag_best = c, float(lag)
    return lag_best


# ------------------------------------------------------------------ one arm on one clip
def stats_for(clip, lengths=LENGTHS) -> dict:
    return {L: window_statistics(clip, length=L, hop=4) for L in lengths}


def joint_stat(clip, arm) -> tuple[float, np.ndarray]:
    best, mask = -np.inf, np.zeros(len(clip), dtype=bool)
    for L in arm["_lengths"]:
        st = window_statistics(clip, length=L, hop=arm["hop"])
        vals = []
        for w in st:
            v = w.get("coherence") if not w.abstained else np.nan
            vals.append((v - arm["_center"][L]) / arm["_scale"][L] if np.isfinite(v) else np.nan)
        fin = [v for v in vals if np.isfinite(v)]
        if fin and max(fin) > best:
            best = float(max(fin))
            mask = np.zeros(len(clip), dtype=bool)
            j = int(np.nanargmax(vals))
            mask[st[j].start:st[j].stop] = True
    return (best if np.isfinite(best) else np.nan), mask


def clip_decision(clip, arm, st, tau) -> dict:
    """Flag, mask, testable and interval metrics for one clip under one arm."""
    n = len(clip)
    kind = arm["kind"]
    if kind == "median":
        stat = clip_median_whole(clip)
        mask = np.zeros(n, dtype=bool)
        if np.isfinite(stat) and np.isfinite(tau) and stat > tau:
            mask[:] = True
        testable = np.isfinite(stat) and np.isfinite(tau)
    elif kind == "bh":
        fit_sorted = arm["_fit_sorted"]
        ps, keep = [], []
        for w in st[arm["length"]]:
            v = w.get(arm["tier"]) if not w.abstained else np.nan
            if np.isfinite(v):
                ps.append(empirical_p(v, fit_sorted))
                keep.append(w)
        rej = bh_reject(ps, arm["q"])
        mask = np.zeros(n, dtype=bool)
        for w, r in zip(keep, rej):
            if r:
                mask[w.start:w.stop] = True
        stat = arm_value(st[arm["length"]], arm)
        testable = bool(keep)
    elif kind == "joint":
        stat, mask = joint_stat(clip, arm)
        testable = np.isfinite(stat)
    elif kind == "lag":
        lag = best_lag(clip)
        shifted = C(clip.lean_dep, _shift_series(clip.lean_req, clip.t, lag),
                    clip.tilt_req_deg, clip.t, clip.truth, clip.yaw, clip.clip_id)
        st2 = {arm["length"]: window_statistics(shifted, length=arm["length"], hop=arm["hop"])}
        stat = arm_value(st2[arm["length"]], arm)
        mask = arm_mask(st2[arm["length"]], arm, tau) if np.isfinite(tau) else np.zeros(n, bool)
        testable = np.isfinite(stat) and np.isfinite(tau)
        clip = shifted
    else:
        stat = arm_value(st[arm["length"]], arm)
        mask = arm_mask(st[arm["length"]], arm, tau) if np.isfinite(tau) else np.zeros(n, bool)
        testable = np.isfinite(stat) and np.isfinite(tau)

    sc = interval_scores(mask, clip.truth)
    sc.update({"flagged": bool(mask.any()), "testable": bool(testable),
               "stat": float(stat) if np.isfinite(stat) else float("nan"),
               "tau": float(tau) if tau is not None and np.isfinite(tau) else float("nan")})
    nseg = len(merge_intervals(mask))
    if clip.truth.any():
        sc["false_segments"] = max(0, nseg - len(merge_intervals(mask & clip.truth)))
    else:
        sc["false_segments"] = nseg
    sc["flagged_frames"] = int(mask.sum())
    sc["truth_frames"] = int(clip.truth.sum())
    sc["_mask"] = mask
    if kind in ("window", "bh", "joint", "median") or np.isfinite(tau):
        if arm.get("refine_sigma") is not None:
            sc = refine_edges(clip, arm, mask, sc)
    return sc


def refine_edges(clip, arm, mask, sc) -> dict:
    if mask is None or not mask.any():
        sc.update({"iou_refined": sc["iou"], "f1_refined": sc["f1"],
                   "onset_refined": sc["onset_err_frames"]})
        return sc
    r_abs = np.abs(frame_residual_deg(clip.lean_dep, clip.lean_req, clip.yaw))
    ext = 2 * arm["length"]
    refined = np.zeros(len(clip), dtype=bool)
    for lo, hi in merge_intervals(mask):
        on, off = refine_region(max(0, lo - ext), min(len(clip), hi + ext), r_abs,
                                arm["refine_sigma"])
        refined[on:off] = True
    s = interval_scores(refined, clip.truth)
    sc.update({"iou_before": sc["iou"], "iou_refined": s["iou"],
               "f1_refined": s["f1"], "onset_refined": s["onset_err_frames"]})
    return sc


# ------------------------------------------------------------------ fit side
def fit_arm(arm, fit_clips: dict, st_fit: dict) -> dict:
    """Everything the arm needs from the FIT population, computed once."""
    kind = arm["kind"]
    if kind == "bh":
        arm["_fit_sorted"] = np.sort(np.concatenate(
            [window_values(st_fit[(c, arm["length"])], arm["tier"]) for c in fit_clips]))
        arm["_tau"] = float("nan")
        return arm
    if kind == "joint":
        arm["_lengths"] = list(LENGTHS)
        arm["_center"], arm["_scale"] = {}, {}
        for L in LENGTHS:
            c0, s0 = robust(np.concatenate([window_values(st_fit[(x, L)], "coherence")
                                            for x in fit_clips]))
            arm["_center"][L], arm["_scale"][L] = c0, s0
        vals = [joint_stat(clip, arm)[0] for clip in fit_clips.values()]
        arm["_tau"] = quantile(vals)
        return arm
    if kind == "median":
        arm["_tau"] = quantile([clip_median_whole(x) for x in fit_clips.values()])
        return arm
    if kind == "lag":
        vals = []
        for x in fit_clips.values():
            lag = best_lag(x)
            shifted = C(x.lean_dep, _shift_series(x.lean_req, x.t, lag), x.tilt_req_deg,
                        x.t, x.truth, x.yaw, x.clip_id)
            st = window_statistics(shifted, length=arm["length"], hop=arm["hop"])
            vals.append(arm_value(st, arm))
        arm["_tau"] = quantile(vals)
        return arm
    arm["_tau"] = quantile([arm_value(st_fit[(c, arm["length"])], arm) for c in fit_clips])
    return arm


def fit_refined(arm, base, fit_clips: dict) -> dict:
    """A3's sigma: 1.4826 * MAD of the fit feasible clips' per-frame |residual|."""
    vals = []
    for x in fit_clips.values():
        r = np.abs(frame_residual_deg(x.lean_dep, x.lean_req, x.yaw))
        vals.append(r[np.isfinite(r)])
    flat = np.concatenate(vals) if vals else np.array([])
    arm["_tau"] = base["_tau"]
    arm["length"] = base["length"]
    arm["hop"] = base["hop"]
    arm["refine_sigma"] = max(1e-3, 1.4826 * float(np.median(flat))) if flat.size else 1e-3
    return arm


# ------------------------------------------------------------------ sim block
def fraction_label(row: dict) -> str:
    v = float(row.get("interval_fraction") or 0.0)
    return "1.0" if v >= 1.0 else f"{v:g}"


def cell_of(row: dict) -> tuple[str, str]:
    frac = fraction_label(row)
    if row.get("defect_family") == "partial":
        return (row.get("violation") or "partial"), frac
    return (row.get("defect_family") or "feasible"), frac


def run_sim(args, arms, out, smoke: bool) -> dict:
    man = read_manifest(args.manifest)
    dev_all = load_split(args.scores_dev, man)
    dev = {c: x for c, x in dev_all.items() if man[c]["split"] == "dev"}
    fit_ids = [c for c in dev if man[c]["label"] == "feasible"]
    if smoke:
        ev_ids = [c for c in dev if man[c]["label"] == "infeasible"]
        ev_scores = args.scores_dev
        print(f"SMOKE on dev: fit {len(fit_ids)} feasible, evaluate {len(ev_ids)} dev defect clips")
    else:
        ev_ids = None
        ev_scores = args.scores_heldout
    ev_all = load_split(ev_scores, man)
    if ev_ids is None:
        ev_ids = [c for c in ev_all if man[c]["split"] == "heldout"]
    ev = {c: ev_all[c] for c in ev_ids if c in ev_all}
    print(f"fit clips {len(fit_ids)}; evaluated clips {len(ev)}")

    fit_clips = {c: dev[c] for c in fit_ids}
    st_fit = {(c, L): window_statistics(fit_clips[c], length=L, hop=4)
              for c in fit_ids for L in LENGTHS}
    st_ev = {(c, L): window_statistics(ev[c], length=L, hop=4) for c in ev for L in LENGTHS}

    for a in arms:
        if a["kind"] == "refined":
            base = next(b for b in arms if b["name"] == "coh12_p1")
            fit_refined(a, base, fit_clips)
        else:
            fit_arm(a, fit_clips, st_fit)

    per = defaultdict(dict)
    base_arm = next(a for a in arms if a["name"] == "coh12_p1")
    for a in arms:
        for c in ev:
            st_clip = {L: st_ev[(c, L)] for L in LENGTHS}
            if a["kind"] == "refined":
                sc = dict(per[base_arm["name"]][c])
                sc = refine_edges(ev[c], a, sc.pop("_mask"), sc)
            else:
                sc = clip_decision(ev[c], a, st_clip, a.get("_tau", float("nan")))
            per[a["name"]][c] = sc

    feas = [c for c in ev if man[c]["label"] == "feasible"]
    base = next(a for a in arms if a["name"] == "median_baseline")
    base_frr = float(np.mean([per["median_baseline"][c]["flagged"] for c in feas])) if feas else float("nan")

    # matched false-reject thresholds, from the FIT side only
    matched = {}
    for a in arms:
        if a["kind"] in ("bh", "joint", "median"):
            continue
        if a["kind"] == "lag":
            vals = []
            for x in fit_clips.values():
                lag = best_lag(x)
                shifted = C(x.lean_dep, _shift_series(x.lean_req, x.t, lag), x.tilt_req_deg,
                            x.t, x.truth, x.yaw, x.clip_id)
                st = window_statistics(shifted, length=a["length"], hop=a["hop"])
                vals.append(arm_value(st, a))
            matched[a["name"]] = quantile(vals, alpha=base_frr) if base_frr > 0 else float("nan")
        else:
            matched[a["name"]] = quantile([arm_value(st_fit[(c, a["length"])], a)
                                           for c in fit_ids],
                                          alpha=base_frr) if base_frr > 0 else float("nan")

    rows = []
    for a in arms:
        for c in ev:
            m = man[c]
            defect, frac = cell_of(m)
            sc = dict(per[a["name"]][c])
            sc.update({"arm": a["name"], "clip_id": c, "scene": m["family"],
                       "defect": defect, "fraction": frac, "label": m["label"],
                       "magnitude_deg": m.get("magnitude_deg"),
                       "uptilt_deg": m.get("uptilt_deg")})
            if a["name"] in matched and np.isfinite(matched[a["name"]]):
                sc["flagged_matched"] = bool(np.isfinite(sc["stat"])
                                             and sc["stat"] > matched[a["name"]])
            sc.pop("_mask", None)
            rows.append(sc)
    dump(rows, out / ("smoke_sim_clips.csv" if smoke else "sim_clips.csv"))

    def cell_rows(arm_name: str, defect: str, frac: str, fam: str = "ALL") -> list[str]:
        return [c for c in ev if cell_of(man[c]) == (defect, frac)
                and (fam == "ALL" or man[c]["family"] == fam)]

    arms_agg = []
    for a in arms:
        for fam in ("field_settlement", "coastal_cliff", "urban_checker", "forest_ridge",
                    "industrial", "ALL"):
            defects = {cell_of(man[c]) for c in ev
                       if fam == "ALL" or man[c]["family"] == fam}
            for defect, frac in sorted(defects):
                ids = cell_rows(a["name"], defect, frac, fam)
                test = [c for c in ids if per[a["name"]][c]["testable"]]
                k = sum(per[a["name"]][c]["flagged"] for c in test)
                km = sum(bool(per[a["name"]][c].get("flagged_matched")) for c in test)
                iou = [per[a["name"]][c]["iou"] if per[a["name"]][c]["flagged"] else 0.0
                       for c in test]
                iou_det = [per[a["name"]][c]["iou"] for c in test if per[a["name"]][c]["flagged"]]
                onset = [per[a["name"]][c]["onset_err_frames"] for c in test
                         if per[a["name"]][c]["flagged"]]
                arms_agg.append({
                    "arm": a["name"], "scene": fam, "defect": defect, "fraction": frac,
                    "n_clips": len(ids), "n_testable": len(test), "k": k,
                    "rate": (k / len(test)) if test else None,
                    "k_matched": km, "rate_matched": (km / len(test)) if test else None,
                    "iou_med": float(np.median(iou)) if iou else None,
                    "iou_med_detected": float(np.median(iou_det)) if iou_det else None,
                    "f1_med": float(np.median([per[a["name"]][c]["f1"] if
                                               per[a["name"]][c]["flagged"] else 0.0
                                               for c in test])) if test else None,
                    "onset_med": float(np.median(onset)) if onset else None,
                    "iou_refined_med": (float(np.median(
                        [per[a["name"]][c].get("iou_refined", np.nan) if
                         per[a["name"]][c]["flagged"] else 0.0 for c in test]))
                        if a["kind"] == "refined" and test else None)})
    dump(arms_agg, out / ("smoke_sim_arms.csv" if smoke else "sim_arms.csv"))

    frr, fdp = [], []
    for a in arms:
        for fam in ("field_settlement", "coastal_cliff", "ALL"):
            ids = [c for c in feas if fam == "ALL" or man[c]["family"] == fam]
            test = [c for c in ids if per[a["name"]][c]["testable"]]
            k = sum(per[a["name"]][c]["flagged"] for c in test)
            mins = sum(len(ev[c]) for c in test) / FPS_SIM / 60.0
            fseg = sum(per[a["name"]][c]["false_segments"] for c in test)
            frr.append({"arm": a["name"], "scene": fam, "n_testable": len(test), "k": k,
                        "rate": (k / len(test)) if test else None,
                        "false_segments": fseg, "minutes": mins,
                        "false_segments_per_min": (fseg / mins) if mins else None})
        fnum = fden = 0
        for c in ev:
            if man[c]["defect_family"] != "partial":
                continue
            sc = per[a["name"]][c]
            tot = int(sc["flagged_frames"])
            if tot == 0:
                continue
            true_pos = int(round(float(sc["recall"]) * int(sc["truth_frames"])))
            fnum += max(0, tot - true_pos)
            fden += tot
        fdp.append({"arm": a["name"], "q": a.get("q") if a["kind"] == "bh" else None,
                    "n_flagged_frames": fden,
                    "realised_fdp": (fnum / fden) if fden else None})
    dump(frr, out / ("smoke_sim_false_reject.csv" if smoke else "sim_false_reject.csv"))
    dump(fdp, out / ("smoke_sim_fdp.csv" if smoke else "sim_fdp.csv"))

    # success criteria, as written
    def agg_lookup(arm, defect, frac, fam="ALL", field="rate"):
        for r in arms_agg:
            if r["arm"] == arm and r["scene"] == fam and r["defect"] == defect \
                    and r["fraction"] == frac:
                return r
        return None

    sc_rows = []
    for defect in ("tilt_removed", "mirrored_roll"):
        p = agg_lookup("coh12_p1", defect, "0.25")
        sc_rows.append({"criterion": "S1", "defect": defect, "fraction": "0.25",
                        "arm": "coh12_p1", "n_clips": p["n_clips"] if p else None,
                        "n_testable": p["n_testable"] if p else None, "k": p["k"] if p else None,
                        "value": p["rate"] if p else None,
                        "iou_med": p["iou_med"] if p else None,
                        "met": bool(p and p["rate"] is not None and p["rate"] >= 0.5
                                    and p["iou_med"] is not None and p["iou_med"] > 0.3)})
        q = agg_lookup("res8_p1", defect, "0.25")
        sc_rows.append({"criterion": "S2", "defect": defect, "fraction": "0.25",
                        "arm": "coh12_p1 - res8_p1 (matched)",
                        "n_clips": p["n_clips"] if p else None,
                        "n_testable": p["n_testable"] if p else None, "k": None,
                        "value": ((p["rate_matched"] - q["rate_matched"])
                                  if p and q and p["rate_matched"] is not None
                                  and q["rate_matched"] is not None else None),
                        "iou_med": None,
                        "met": bool(p and q and p["rate_matched"] is not None
                                    and q["rate_matched"] is not None
                                    and (p["rate_matched"] - q["rate_matched"]) >= 0.10)})
    f2 = next(r for r in frr if r["arm"] == "a2_bh_coh12_q10" and r["scene"] == "ALL")
    f1 = next(r for r in frr if r["arm"] == "coh12_p1" and r["scene"] == "ALL")
    fd = next(r for r in fdp if r["arm"] == "a2_bh_coh12_q10")
    sc_rows.append({"criterion": "S3", "defect": "partial (any)", "fraction": "all",
                    "arm": "a2_bh_coh12_q10", "n_clips": None, "n_testable": f2["n_testable"],
                    "k": f2["k"], "value": fd["realised_fdp"], "iou_med": None,
                    "met": bool(fd["realised_fdp"] is not None and fd["realised_fdp"] <= 0.20
                                and f2["false_segments_per_min"] is not None
                                and f1["false_segments_per_min"] is not None
                                and (f2["false_segments_per_min"]
                                     - f1["false_segments_per_min"]) <= 0.5)})
    # S4-S6 from the refined / lag / joint arms against their comparators
    def iou_delta(arm, base_arm):
        pairs = []
        for c in ev:
            if man[c]["defect_family"] == "feasible":
                continue
            a = per[arm][c]
            if not a["flagged"]:
                continue
            if "iou_refined" in a:
                pairs.append((a["iou"], a["iou_refined"]))
        if not pairs:
            return None, None
        d = [b - a for a, b in pairs]
        return float(np.median(d)), len(d)

    d4, n4 = iou_delta("a3_refined", "coh12_p1")
    sc_rows.append({"criterion": "S4", "defect": "any flagged", "fraction": "all",
                    "arm": "a3_refined - coh12_p1", "n_clips": n4, "n_testable": n4,
                    "k": None, "value": d4, "iou_med": None,
                    "met": bool(d4 is not None and d4 >= 0.05)})
    lag_arm = next(a for a in arms if a["kind"] == "lag")
    fr_lag = next(r for r in frr if r["arm"] == lag_arm["name"] and r["scene"] == "ALL")
    d5 = (1.0 - fr_lag["false_segments_per_min"] / f1["false_segments_per_min"]
          if fr_lag["false_segments_per_min"] and f1["false_segments_per_min"] else None)
    iou_lag = next(r for r in arms_agg if r["arm"] == lag_arm["name"]
                   and r["scene"] == "ALL" and r["defect"] == "tilt_removed"
                   and r["fraction"] == "1.0")
    iou_p1 = next(r for r in arms_agg if r["arm"] == "coh12_p1" and r["scene"] == "ALL"
                  and r["defect"] == "tilt_removed" and r["fraction"] == "1.0")
    d5iou = ((iou_lag["iou_med"] - iou_p1["iou_med"])
             if iou_lag and iou_p1 and iou_lag["iou_med"] is not None
             and iou_p1["iou_med"] is not None else None)
    sc_rows.append({"criterion": "S5", "defect": "all", "fraction": "all",
                    "arm": f"{lag_arm['name']} - coh12_p1", "n_clips": None,
                    "n_testable": fr_lag["n_testable"], "k": None,
                    "value": d5, "iou_med": d5iou,
                    "met": bool((d5 is not None and d5 >= 0.20) or
                                (d5iou is not None and d5iou >= 0.05))})
    best_single = None
    for name in ("coh12_p1", "coh8_p1", "res12_p1", "res8_p1", "res8_p2", "sign12_p1"):
        for defect in ("tilt_removed", "mirrored_roll"):
            r = agg_lookup(name, defect, "0.25", field="rate_matched")
            if r and r["rate_matched"] is not None:
                if best_single is None or r["rate_matched"] > best_single:
                    best_single = r["rate_matched"]
    j = agg_lookup("a5_joint", "tilt_removed", "0.25", field="rate_matched")
    j2 = agg_lookup("a5_joint", "mirrored_roll", "0.25", field="rate_matched")
    vals = [x["rate_matched"] for x in (j, j2) if x and x["rate_matched"] is not None]
    sc_rows.append({"criterion": "S6", "defect": "tilt_removed+mirrored_roll", "fraction": "0.25",
                    "arm": "a5_joint - best single length", "n_clips": None,
                    "n_testable": (j["n_testable"] if j else None), "k": None,
                    "value": ((float(np.mean(vals)) - best_single)
                              if vals and best_single is not None else None),
                    "iou_med": None,
                    "met": bool(vals and best_single is not None
                                and (float(np.mean(vals)) - best_single) >= 0.05)})
    dump(sc_rows, out / ("smoke_success_criteria.csv" if smoke else "success_criteria.csv"))

    summary = {
        "mode": "SMOKE-DEV" if smoke else "held-out", "frozen": str(FROZEN),
        "alpha": ALPHA, "n_fit_feasible": len(fit_ids), "n_evaluated": len(ev),
        "n_feasible_evaluated": len(feas), "baseline_frr": base_frr,
        "taus": {a["name"]: a.get("_tau") for a in arms},
        "matched_taus": matched,
        "n_by_cell": {f"{d}|{f}": len([c for c in ev if cell_of(man[c]) == (d, f)])
                      for d, f in sorted({cell_of(man[c]) for c in ev})},
        "n_by_cell_scene": {f"{sc}|{d}|{f}": len([c for c in ev if cell_of(man[c]) == (d, f)
                                                  and man[c]["family"] == sc])
                            for d, f in sorted({cell_of(man[c]) for c in ev})
                            for sc in sorted({man[c]["family"] for c in ev})},
        "arm_kinds": {a["name"]: a["kind"] for a in arms},
    }
    (out / ("smoke_summary.json" if smoke else "sim_summary.json")).write_text(
        json.dumps(summary, indent=1, default=float))
    return {"arms_agg": arms_agg, "frr": frr, "fdp": fdp, "sc": sc_rows, "summary": summary,
            "per": per, "ev": ev, "man": man}


# ------------------------------------------------------------------ real block
def load_real_clip(real_root: Path, clip_id: str, cond: str):
    from src.verdict.schema import load_clip

    flight = clip_id.split("#")[0]
    return load_clip(real_root / flight / f"{clip_id}__{cond}")


def real_condition_truth(cond: str, n: int) -> np.ndarray:
    if cond in REAL_CONTROLS or cond == "real":
        return np.zeros(n, dtype=bool)
    return np.ones(n, dtype=bool)


def real_compose(real_root: Path, clip_id: str, cond: str, frac: float,
                 real) -> tuple[object, np.ndarray]:
    edit = load_real_clip(real_root, clip_id, cond)
    n = len(real.t)
    if frac >= 1.0:
        return edit, real_condition_truth(cond, n)
    m = max(1, int(round(frac * n)))
    lo = (n - m) // 2
    mask = np.zeros(n, dtype=bool)
    mask[lo:lo + m] = True
    dep = real.depicted_lean.copy()
    dep[mask] = edit.depicted_lean[mask]
    # controls are not violations by construction: their truth mask is empty
    truth = np.zeros(n, dtype=bool) if cond in REAL_CONTROLS else mask
    return replace(real, depicted_lean=dep), truth


def run_real(arms, out, smoke: bool, real_root: Path) -> dict:
    from src.verdict.abstain import gate_kept_from_conf
    from src.verdict.schema import load_clip

    gate_cfg = yaml.safe_load(GATE_CFG.read_text())

    def population(flights):
        clips, gate = {}, {}
        for f in flights:
            for d in sorted((real_root / f).iterdir()):
                if d.name.endswith("__real"):
                    cid = d.name[: -len("__real")]
                    real = load_clip(d)
                    clips[cid] = real
                    gate[cid] = gate_kept_from_conf(real.depicted_conf, gate_cfg)
        return clips, gate

    dev_clips, dev_gate = population(DEV_REAL_FLIGHTS)
    held_clips, held_gate = population(["flight-13p-trackRATM", "flight-07a-lemniscate"])
    fit_ids = [c for c in dev_clips if dev_gate[c]]
    print(f"real dev gate-kept fit clips {len(fit_ids)}; held-out clips {len(held_clips)} "
          f"(gate-kept {sum(held_gate.values())})")

    arms_r = [a for a in arms if a["kind"] in ("window", "median", "joint")]
    per_arm_tau = {}
    for a in arms_r:
        if a["kind"] == "median":
            per_arm_tau[a["name"]] = quantile([clip_median_whole(dev_clips[c]) for c in fit_ids])
        elif a["kind"] == "joint":
            for L in LENGTHS:
                vals = []
                for c in fit_ids:
                    vals.extend(window_values(
                        window_statistics(dev_clips[c], length=L, hop=4), "coherence"))
                a.setdefault("_center", {})[L], a.setdefault("_scale", {})[L] = robust(vals)
            a["_lengths"] = list(LENGTHS)
            vals = [joint_stat(wrap_real(dev_clips[c]), a)[0] for c in fit_ids]
            per_arm_tau[a["name"]] = quantile(vals)
        else:
            vals = [arm_value(window_statistics(dev_clips[c], length=a["length"], hop=a["hop"]), a)
                    for c in fit_ids]
            per_arm_tau[a["name"]] = quantile(vals)

    rows = []
    for cid, real in held_clips.items():
        if not held_gate[cid]:
            continue
        for a in arms_r:
            tau = per_arm_tau[a["name"]]
            for frac in REAL_FRACTIONS:
                for cond in ("real",) + REAL_FAMILIES + REAL_CONTROLS:
                    if cond == "real" and frac != 1.0:
                        continue
                    if cond == "real":
                        clip, truth = real, np.zeros(len(real.t), bool)
                        label = "real"
                    else:
                        clip, truth = real_compose(real_root, cid, cond, frac, real)
                        label = cond if frac >= 1.0 else f"{cond}@{frac:g}"
                    wrapped = wrap_real(clip, truth)
                    if a["kind"] == "median":
                        sc = clip_decision(wrapped, a, {}, tau)
                    elif a["kind"] == "joint":
                        sc = clip_decision(wrapped, a, {}, tau)
                    else:
                        st = {a["length"]: window_statistics(wrapped, length=a["length"], hop=a["hop"])}
                        sc = clip_decision(wrapped, a, st, tau)
                    rows.append({"arm": a["name"], "clip_id": cid, "condition": label,
                                 "testable": sc["testable"], "flagged": sc["flagged"],
                                 "stat": sc["stat"], "tau": tau, "iou": sc["iou"],
                                 "f1": sc["f1"], "onset": sc["onset_err_frames"]})
    dump(rows, out / ("smoke_real_clips.csv" if smoke else "real_clips.csv"))
    agg = []
    for a in arms_r:
        for cond in ("real",) + tuple(f"{c}@{f:g}" for f in REAL_FRACTIONS
                                      for c in REAL_FAMILIES + REAL_CONTROLS) + \
                REAL_FAMILIES + REAL_CONTROLS:
            test = [r for r in rows if r["arm"] == a["name"] and r["condition"] == cond
                    and r["testable"]]
            k = sum(r["flagged"] for r in test)
            agg.append({"arm": a["name"], "condition": cond, "n_testable": len(test), "k": k,
                        "rate": (k / len(test)) if test else None,
                        "iou_med": (float(np.median([r["iou"] if r["flagged"] else 0.0
                                                     for r in test])) if test else None)})
    dump(agg, out / ("smoke_real_arms.csv" if smoke else "real_arms.csv"))
    return {"rows": rows, "agg": agg, "taus": per_arm_tau, "fit_ids": fit_ids}


def wrap_real(clip, truth=None):
    yaw = np.asarray(clip.yaw, float)
    tr = np.asarray(clip.tilt_req_deg, float)
    truth = np.zeros(len(clip.t), bool) if truth is None else truth
    return C(clip.depicted_lean, clip.lean_req, tr, clip.t, truth, yaw,
             getattr(clip, "clip_id", ""))


# ------------------------------------------------------------------ io
def dump(rows: list[dict], path: Path) -> None:
    """Ordered union of the rows' keys, so arms with extra fields (A3) write cleanly."""
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dev", default="results/g06/scores")
    ap.add_argument("--scores-heldout", default="results/g07/scores")
    ap.add_argument("--manifest",
                    default="/home/kevin/Documents/Projects/awmv-track-i/results/i02/manifest.csv")
    ap.add_argument("--out", default="results/g07")
    ap.add_argument("--real-clips",
                    default=str(Path.home() / "Documents/Projects/aerial-world-model-verifier"
                                / "data/processed/track_d/clips"),
                    help="schema-v2 real clips (Track D); this worktree has no data/ link")
    ap.add_argument("--smoke-dev", action="store_true",
                    help="code check: fit and evaluate on DEV clips only, write smoke_* files")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = load_arms()
    sim = run_sim(args, arms, out, args.smoke_dev)
    real = run_real(arms, out, args.smoke_dev, Path(args.real_clips))

    print("\n(sim) primary arm vs baseline by cell: n_clips | testable | k (primary) | k (baseline)")
    for r in sim["arms_agg"]:
        if r["scene"] != "ALL" or r["arm"] not in ("coh12_p1", "median_baseline"):
            continue
        print(f"  {r['arm']:16s} {r['defect']:24s} {r['fraction']:4s} n={r['n_clips']:3d} "
              f"test={r['n_testable']:3d} k={r['k']:3d} rate={r['rate']}")
    print("\n(real) condition rates, primary arm:")
    for r in real["agg"]:
        if r["arm"] == "coh12_p1" and r["condition"] in ("real",) + REAL_FAMILIES + REAL_CONTROLS:
            print(f"  {r['condition']:16s} {r['k']}/{r['n_testable']} rate={r['rate']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
