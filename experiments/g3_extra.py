"""Track G3: ablations A2-A5 and the minimum-detectable-violation curve.

All arms, thresholds and grids are the pre-registered ones in
``configs/g06_segments_prereg.yaml``.  Runs on development clips only; nothing here touches
held-out data.

    pixi run python experiments/g3_extra.py
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

from experiments.g3_ablation import (ALPHA, ARMS, BASE, arm_mask, arm_value,   # noqa: E402
                                     quantile, x_frac)
from experiments.g3_data import load_split, read_manifest                      # noqa: E402
from src.verdict.fit import _shift_series                                     # noqa: E402
from src.verdict.segments import (clip_max, clip_median_whole,                # noqa: E402
                                  flagged_windows, frame_residual_deg,
                                  interval_scores, merge_intervals, window_statistics)

ARM = {a["name"]: a for a in ARMS}
PRIMARY = ARM["coh12_p1"]
COMPARE = ARM["res8_p1"]


# ------------------------------------------------------------------ shared prep
def prepare(args):
    man = read_manifest(args.manifest)
    clips = load_split(args.scores, man)
    dev = {c: x for c, x in clips.items() if man.get(c, {}).get("split") == "dev"}
    stats = {}
    for length in (8, 12, 16):
        for c, x in dev.items():
            stats[(c, length)] = window_statistics(x, length=length, hop=4)
    seeds = sorted({int(man[c]["scene_seed"]) for c in dev})
    feasible = [c for c in dev if man[c]["label"] == "feasible"]
    return man, dev, stats, seeds, feasible


def fold_tau(values, man, feasible, seeds, seed, alpha=ALPHA):
    fit = [c for c in feasible if int(man[c]["scene_seed"]) != seed]
    return quantile([values[c] for c in fit], alpha)


def window_values(stats, tier):
    out = []
    for s in stats:
        if not s.abstained:
            v = s.get(tier)
            if np.isfinite(v):
                out.append(v)
    return np.array(out)


def bh_reject(pvals, q):
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(m, bool)
    k = np.max(np.flatnonzero(passed)) + 1
    mask = np.zeros(m, bool)
    mask[order[:k]] = True
    return mask


def empirical_p(w, fit_sorted):
    if len(fit_sorted) == 0:
        return np.nan
    ge = len(fit_sorted) - np.searchsorted(fit_sorted, w, side="left")
    return (1.0 + ge) / (len(fit_sorted) + 1.0)


# ------------------------------------------------------------------ A2 FDR
def run_fdr_rates(man, dev, stats, seeds, feasible, out):
    """Rate table (detection per cell) and realised FDP per arm/q/mode."""
    rows, fdp_rows = [], []
    for arm in (PRIMARY, COMPARE):
        tier, length = arm["tier"], arm["length"]
        for q in (0.05, 0.10, 0.20):
            for mode in ("per_clip", "corpus"):
                det = defaultdict(lambda: [0, 0])
                fnum = fden = 0
                fseg = minutes = 0.0
                for seed in seeds:
                    fit = [c for c in feasible if int(man[c]["scene_seed"]) != seed]
                    fit_vals = np.sort(np.concatenate([window_values(stats[(c, length)], tier)
                                                       for c in fit])) if fit else np.array([])
                    fold_clips = [c for c in dev if int(man[c]["scene_seed"]) == seed]
                    if mode == "corpus":
                        keys, ps = [], []
                        for c in fold_clips:
                            for j, s in enumerate(stats[(c, length)]):
                                v = s.get(tier) if not s.abstained else np.nan
                                if np.isfinite(v):
                                    keys.append((c, j))
                                    ps.append(empirical_p(v, fit_vals))
                        rej = bh_reject(ps, q)
                        rejected = {k for k, r in zip(keys, rej) if r}
                        for c in fold_clips:
                            mask = np.zeros(len(dev[c]), dtype=bool)
                            for j, s in enumerate(stats[(c, length)]):
                                if (c, j) in rejected:
                                    mask[s.start:s.stop] = True
                            _acc(man, dev, c, mask, det, lambda a, b: None)
                            if dev[c].truth.any() and mask.any():
                                fnum += int(mask.sum() - (mask & dev[c].truth).sum())
                                fden += int(mask.sum())
                            elif not dev[c].truth.any():
                                fseg += len(merge_intervals(mask))
                                minutes += len(dev[c]) / 30.0 / 60.0
                    else:
                        for c in fold_clips:
                            ps, ss = [], []
                            for s in stats[(c, length)]:
                                v = s.get(tier) if not s.abstained else np.nan
                                if np.isfinite(v):
                                    ps.append(empirical_p(v, fit_vals))
                                    ss.append(s)
                            rej = bh_reject(ps, q)
                            mask = np.zeros(len(dev[c]), dtype=bool)
                            for s, r in zip(ss, rej):
                                if r:
                                    mask[s.start:s.stop] = True
                            _acc(man, dev, c, mask, det, lambda a, b: None)
                            if dev[c].truth.any() and mask.any():
                                fnum += int(mask.sum() - (mask & dev[c].truth).sum())
                                fden += int(mask.sum())
                            elif not dev[c].truth.any():
                                fseg += len(merge_intervals(mask))
                                minutes += len(dev[c]) / 30.0 / 60.0
                for (fam, tgt, frac), (k, n) in sorted(det.items()):
                    rows.append({"arm": arm["name"], "q": q, "mode": mode, "family": fam,
                                 "defect": tgt, "fraction": frac, "k": k, "n": n,
                                 "rate": k / n if n else None})
                fdp_rows.append({"arm": arm["name"], "q": q, "mode": mode,
                                 "fdp": (fnum / fden) if fden else None,
                                 "windows_outside": fnum, "windows_flagged": fden,
                                 "false_segments": fseg, "minutes": minutes,
                                 "false_segments_per_min": (fseg / minutes) if minutes else None})
    dump(rows, out / "a2_fdr_rates.csv")
    dump(fdp_rows, out / "a2_fdr_realised.csv")
    return rows, fdp_rows


def _acc(man, dev, c, mask, det, _):
    if not dev[c].truth.any():
        return
    key = (man[c]["family"], man[c]["defect_family"], f"{x_frac(man[c]):g}")
    det[key][1] += 1
    det[key][0] += int(bool(mask.any()))


# ------------------------------------------------------------------ A3 boundaries
def _logpdf_state0(x, sigma):
    return -0.5 * (x / sigma) ** 2


def _logpdf_state1(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2


def refine_region(a, b, r_abs, sigma, max_mu=None):
    n = len(r_abs)
    a, b = max(0, a), min(n, b)
    best = (-np.inf, a, b)
    for on in range(a, b + 1):
        for off in range(on + 1, b + 1):
            inside = r_abs[on:off]
            mu = float(np.mean(inside)) if len(inside) else 0.0
            ll = (np.sum(_logpdf_state1(inside, mu, sigma))
                  + np.sum(_logpdf_state0(np.concatenate([r_abs[a:on], r_abs[off:b]]), sigma)))
            if ll > best[0]:
                best = (ll, on, off)
    return best[1], best[2]


def run_boundaries(man, dev, stats, seeds, feasible, out, extend_windows=2):
    sigma = float(np.median(np.abs(np.concatenate(
        [frame_residual_deg(dev[c].lean_dep, dev[c].lean_req, dev[c].yaw) for c in feasible]))))
    sigma = max(1e-3, 1.4826 * sigma)
    rows = []
    for c, x in dev.items():
        if not x.truth.any():
            continue
        tau = fold_tau({cc: arm_value(stats[(cc, PRIMARY["length"])], PRIMARY) for cc in dev},
                       man, feasible, seeds, int(man[c]["scene_seed"]))
        st = stats[(c, PRIMARY["length"])]
        mask = arm_mask(st, PRIMARY, tau) if np.isfinite(tau) else np.zeros(len(x), bool)
        if not mask.any():
            continue
        r_abs = np.abs(frame_residual_deg(x.lean_dep, x.lean_req, x.yaw))
        ext = extend_windows * primary_len()
        refined = np.zeros(len(x), bool)
        for lo, hi in merge_intervals(mask):
            a, b = max(0, lo - ext), min(len(x), hi + ext)
            on, off = refine_region(a, b, r_abs, sigma)
            refined[on:off] = True
        s0, s1 = interval_scores(mask, x.truth), interval_scores(refined, x.truth)
        rows.append({"clip_id": c, "family": man[c]["family"], "defect": man[c]["defect_family"],
                     "fraction": f"{x_frac(man[c]):g}",
                     "iou_before": s0["iou"], "iou_after": s1["iou"],
                     "f1_before": s0["f1"], "f1_after": s1["f1"],
                     "onset_before": s0["onset_err_frames"], "onset_after": s1["onset_err_frames"]})
    dump(rows, out / "a3_boundaries.csv")
    return rows


def primary_len():
    return PRIMARY["length"]


# ------------------------------------------------------------------ A4 alignment
def run_alignment(man, dev, seeds, feasible, out):
    lags_s = np.arange(-0.150, 0.1501, 0.025)
    rows, lag_rows = [], []
    for c, x in dev.items():
        dep = np.linalg.norm(x.lean_dep, axis=1)
        best, best_lag = -2.0, 0.0
        for lag in lags_s:
            req = _shift_series(x.lean_req, x.t, float(lag))
            rq = np.linalg.norm(req, axis=1)
            ok = np.isfinite(dep) & np.isfinite(rq)
            if ok.sum() < 5 or np.std(dep[ok]) < 1e-9 or np.std(rq[ok]) < 1e-9:
                continue
            cc = float(np.corrcoef(dep[ok], rq[ok])[0, 1])
            if cc > best:
                best, best_lag = cc, float(lag)
        lag_rows.append({"clip_id": c, "family": man[c]["family"],
                         "uptilt": man[c]["uptilt_deg"], "lag_s": best_lag,
                         "corr": best if best > -2 else None})
    dump(lag_rows, out / "a4_lags.csv")

    # rescore the primary arm with and without the per-clip lag
    def score_all(use_lag):
        vals, masks = {}, {}
        for c, x in dev.items():
            lag = 0.0
            if use_lag:
                lag = next(r["lag_s"] for r in lag_rows if r["clip_id"] == c)
            tgt = x
            if lag:
                req = _shift_series(x.lean_req, x.t, lag)
                tgt = type(x)(clip_id=x.clip_id, t=x.t, lean_dep=x.lean_dep, lean_req=req,
                              tilt_req_deg=np.degrees(np.arcsin(np.clip(
                                  np.linalg.norm(req, axis=1), 0, 1))),
                              truth=x.truth, meta=x.meta)
            st = window_statistics(tgt, length=PRIMARY["length"], hop=4)
            vals[c] = arm_value(st, PRIMARY)
            masks[c] = st
        return vals, masks

    v0, st0 = score_all(False)
    v1, st1 = score_all(True)
    for tag, vals, stmap in (("no_lag", v0, st0), ("lag_removed", v1, st1)):
        det = defaultdict(lambda: [0, 0])
        fseg = minutes = 0.0
        ious = []
        for c, x in dev.items():
            tau = fold_tau(vals, man, feasible, seeds, int(man[c]["scene_seed"]))
            mask = arm_mask(stmap[c], PRIMARY, tau) if np.isfinite(tau) else np.zeros(len(x), bool)
            if x.truth.any():
                s = interval_scores(mask, x.truth)
                key = (man[c]["family"], man[c]["defect_family"], f"{x_frac(man[c]):g}")
                det[key][0] += int(bool(mask.any()))
                det[key][1] += 1
                ious.append(s["iou"] if mask.any() else 0.0)
            else:
                fseg += len(merge_intervals(mask))
                minutes += len(x) / 30.0 / 60.0
        for (fam, tgt, frac), (k, n) in sorted(det.items()):
            rows.append({"arm": tag, "family": fam, "defect": tgt, "fraction": frac,
                         "k": k, "n": n, "rate": k / n if n else None})
        rows.append({"arm": tag, "family": "ALL", "defect": "false_segments",
                     "fraction": "", "k": fseg, "n": minutes, "rate": fseg / minutes if minutes else None})
        rows.append({"arm": tag, "family": "ALL", "defect": "mean_iou", "fraction": "",
                     "k": float(np.mean(ious)) if ious else None, "n": len(ious), "rate": None})
    dump(rows, out / "a4_alignment.csv")
    return rows


# ------------------------------------------------------------------ A5 multiscale
def run_multiscale(man, dev, stats, seeds, feasible, out):
    rows = []
    arm_by_len = {}
    for length in (8, 12, 16):
        a = {"name": f"coh{length}_p1", "tier": "coherence", "length": length,
             "hop": 4, "persistence": 1}
        arm_by_len[length] = a
    vals = {length: {c: arm_value(stats[(c, length)], a) for c in dev}
            for length, a in arm_by_len.items()}
    # standardise on the fit clips per fold, then max over lengths
    joint = {}
    for c in dev:
        seed = int(man[c]["scene_seed"])
        z = []
        for length, a in arm_by_len.items():
            fit = [cc for cc in feasible if int(man[cc]["scene_seed"]) != seed]
            fv = np.array([vals[length][cc] for cc in fit], float)
            fv = fv[np.isfinite(fv)]
            med = np.median(fv) if len(fv) else 0.0
            scale = (np.quantile(fv, 0.975) - np.quantile(fv, 0.025)) / 3.92 if len(fv) > 8 else 1.0
            z.append((vals[length][c] - med) / max(scale, 1e-9))
        joint[c] = float(np.nanmax(z))
    for name, vv in [("joint", joint)] + [(f"coh{L}", vals[L]) for L in (8, 12, 16)]:
        det = defaultdict(lambda: [0, 0])
        ious = []
        fseg = minutes = 0.0
        for c, x in dev.items():
            tau = fold_tau(vv, man, feasible, seeds, int(man[c]["scene_seed"])) \
                if name == "joint" else None
            if name == "joint":
                mask = np.zeros(len(x), bool)
                if np.isfinite(tau) and np.isfinite(vv[c]) and vv[c] > tau:
                    # segments from the winning length
                    seed = int(man[c]["scene_seed"])
                    best, bestz = None, -np.inf
                    for length in (8, 12, 16):
                        fit = [cc for cc in feasible if int(man[cc]["scene_seed"]) != seed]
                        fv = np.array([vals[length][cc] for cc in fit], float)
                        fv = fv[np.isfinite(fv)]
                        med = np.median(fv) if len(fv) else 0.0
                        scale = (np.quantile(fv, 0.975) - np.quantile(fv, 0.025)) / 3.92 \
                            if len(fv) > 8 else 1.0
                        zz = (vals[length][c] - med) / max(scale, 1e-9)
                        if zz > bestz:
                            bestz, best = zz, length
                    ttau = quantile([vals[best][cc] for cc in feasible
                                     if int(man[cc]["scene_seed"]) != seed])
                    mask = arm_mask(stats[(c, best)], arm_by_len[best], ttau) \
                        if np.isfinite(ttau) else mask
            else:
                a = arm_by_len[int(name[3:])]
                ttau = fold_tau(vv, man, feasible, seeds, int(man[c]["scene_seed"]))
                mask = arm_mask(stats[(c, a["length"])], a, ttau) \
                    if np.isfinite(ttau) else np.zeros(len(x), bool)
            if x.truth.any():
                s = interval_scores(mask, x.truth)
                key = (man[c]["family"], man[c]["defect_family"], f"{x_frac(man[c]):g}")
                det[key][0] += int(bool(mask.any()))
                det[key][1] += 1
                ious.append(s["iou"] if mask.any() else 0.0)
            else:
                fseg += len(merge_intervals(mask))
                minutes += len(x) / 30.0 / 60.0
        for (fam, tgt, frac), (k, n) in sorted(det.items()):
            rows.append({"arm": name, "family": fam, "defect": tgt, "fraction": frac,
                         "k": k, "n": n, "rate": k / n if n else None})
        rows.append({"arm": name, "family": "ALL", "defect": "false_segments", "fraction": "",
                     "k": fseg, "n": minutes, "rate": fseg / minutes if minutes else None})
        rows.append({"arm": name, "family": "ALL", "defect": "mean_iou", "fraction": "",
                     "k": float(np.mean(ious)) if ious else None, "n": len(ious), "rate": None})
    dump(rows, out / "a5_multiscale.csv")
    return rows


# ------------------------------------------------------------------ minimum detectable step
def run_mdv(man, dev, stats, seeds, feasible, out,
            deltas=(2, 4, 6, 8, 10, 12, 14, 16, 20, 24), durations=(0.4, 0.8, 1.6, 3.2)):
    rows = []
    for arm in (PRIMARY, COMPARE):
        rows += _mdv_arm(man, dev, stats, seeds, feasible, arm,
                         deltas=deltas, durations=durations)
    dump(rows, out / "mdv.csv")
    return rows


def _mdv_arm(man, dev, stats, seeds, feasible, arm, deltas, durations):
    rows = []
    vals = {c: arm_value(stats[(c, arm["length"])], arm) for c in dev}
    # noise floor: the reader's run-to-run envelope converted to degrees on a 12-frame window
    for D in durations:
        for delta in deltas:
            k = n = 0
            ious = []
            for c in feasible:
                x = dev[c]
                if len(x) < int(D * 30) + 5:
                    continue
                dirv = np.mean(x.lean_req, axis=0)
                norm = np.linalg.norm(dirv)
                u = dirv / norm if norm > 1e-9 else np.array([1.0, 0.0])
                m = int(round(D * 30))
                start = (len(x) - m) // 2
                lean = x.lean_dep.copy()
                lean[start:start + m] += np.tan(np.radians(delta)) * u
                injected = type(x)(clip_id=x.clip_id, t=x.t, lean_dep=lean,
                                   lean_req=x.lean_req, tilt_req_deg=x.tilt_req_deg,
                                   truth=np.zeros(len(x), bool), meta=x.meta)
                st = window_statistics(injected, length=arm["length"], hop=4)
                tau = fold_tau(vals, man, feasible, seeds, int(man[c]["scene_seed"]))
                mask = arm_mask(st, arm, tau) if np.isfinite(tau) else np.zeros(len(x), bool)
                truth = np.zeros(len(x), bool)
                truth[start:start + m] = True
                s = interval_scores(mask, truth)
                k += int(bool(mask.any()))
                n += 1
                ious.append(s["iou"] if mask.any() else 0.0)
            rows.append({"arm": arm["name"], "duration_s": D, "delta_deg": delta, "k": k, "n": n,
                         "rate": k / n if n else None,
                         "iou_med": float(np.median(ious)) if ious else None,
                         "detected_and_iou_gt_0_3": int(sum(1 for x in ious if x > 0.3)),
                         "rate_iou_gt_0_3": (sum(1 for x in ious if x > 0.3) / n) if n else None})
    return rows


def dump(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/g06/scores")
    ap.add_argument("--manifest",
                    default="/home/kevin/Documents/Projects/awmv-track-i/results/i02/manifest.csv")
    ap.add_argument("--out", default="results/g06/ablation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man, dev, stats, seeds, feasible = prepare(args)
    print(f"{len(dev)} dev clips, {len(feasible)} feasible, seeds {seeds}")
    run_fdr_rates(man, dev, stats, seeds, feasible, out)
    print("A2 done")
    run_boundaries(man, dev, stats, seeds, feasible, out)
    print("A3 done")
    run_alignment(man, dev, seeds, feasible, out)
    print("A4 done")
    run_multiscale(man, dev, stats, seeds, feasible, out)
    print("A5 done")
    run_mdv(man, dev, stats, seeds, feasible, out)
    print("MDV done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
