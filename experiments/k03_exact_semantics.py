"""Track K, Part 3: re-derive J03's absorption and J02's floor under exact drag semantics.

Pre-registered in ``configs/k03_exact_semantics_prereg.yaml`` (commit 3110976).  The only
change from J03/J02 is the required side: the exact flatness axis at a physical drag
coefficient, with no `(k_d - 0.295)` rescaling.  CPU only.

    PYTHONPATH=. python experiments/k03_exact_semantics.py --part all
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.k02_downstream import (CONTROLS, COVERAGES, DEV, FAMILIES, HELDOUT,
                                        K0, WIN, compose_partial, dct_split, dep_axis,
                                        exact_v2_axis, gate_keep, load_clips, make_error,
                                        tier_values, coherence_stat, _jsonable)
from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
from src.airframe.models import axis_of_lean, lean_of_axis, unit
from src.airframe.schema_v3 import sample_axis
from src.physics.flatness import G0
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/k03"
GRID = np.round(np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.295, 0.35, 0.40]), 3)
HUBER_DEG = 3.0
HUBER_DELTA = float(np.sin(np.deg2rad(HUBER_DEG)))   # Huber threshold in lean units
DELTAS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])
J02_DRAWS, GRID_DRAWS, ANCHOR_DRAWS = 120, 60, 120
SIGMA, BIAS, TAU_S, FAST_SHARE = 4.8, 2.0, 0.5, 0.5
TILT_MASK = 85.0


# --------------------------------------------------------------------------- exact sides
def build_sides(data: dict, kin: dict) -> None:
    """Attach exact axis/lean/tilt at every k_d in GRID to every clip."""
    native = {}
    for f, k in kin.items():
        for kd in GRID:
            native[(f, float(kd))] = unit(k.a + np.array([0.0, 0.0, G0]) + kd * k.v)
    for b in data.values():
        f = b["flight"]
        b["sides"] = {}
        for kd in GRID:
            ax, lean, tilt = sample_axis(kin[f], native[(f, float(kd))], b["t"], b["yaw"])
            b["sides"][float(kd)] = {"axis": ax, "lean": lean, "tilt": tilt,
                                     "upright": tilt < TILT_MASK}


def build_sides_linear(data: dict) -> None:
    """J03's construction, for pipeline validation only: lean_req + (k_d - 0.295) vel_req."""
    for b in data.values():
        b["sides"] = {}
        for kd in GRID:
            lean = b["lean_req_v2"] + (kd - K0) * b["vel_req_v2"]
            ax = axis_of_lean(lean, b["yaw"])
            tilt = np.degrees(np.arccos(np.clip(ax[:, 2], -1.0, 1.0)))
            b["sides"][float(kd)] = {"axis": ax, "lean": lean, "tilt": tilt,
                                     "upright": np.linalg.norm(lean, axis=1) < np.sin(
                                         np.deg2rad(TILT_MASK))}


# --------------------------------------------------------------------------- Huber fit
def huber_offset(dep: np.ndarray, req: np.ndarray, delta: float = HUBER_DELTA,
                 iters: int = 6) -> np.ndarray:
    r = dep - req
    ok = np.isfinite(r).all(1)
    if ok.sum() < 4:
        return np.zeros(2)
    r = r[ok]
    o = np.median(r, axis=0)
    for _ in range(iters):
        e = np.linalg.norm(r - o, axis=1)
        w = np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))
        o = np.sum(w[:, None] * r, axis=0) / np.sum(w)
    return o


def huber_loss(dep: np.ndarray, req: np.ndarray) -> float:
    o = huber_offset(dep, req)
    e = np.linalg.norm(dep - req - o, axis=1)
    e = e[np.isfinite(e)]
    d = HUBER_DELTA
    h = np.where(e <= d, 0.5 * e ** 2, d * (e - 0.5 * d))
    return float(np.mean(h)) if len(h) else np.inf


def fit_offset(dep: np.ndarray, b: dict, kd: float) -> np.ndarray:
    return huber_offset(dep, b["sides"][float(kd)]["lean"])


def fit_kd(dep: np.ndarray, b: dict) -> tuple[float, np.ndarray]:
    best = None
    for kd in GRID:
        loss = huber_loss(dep, b["sides"][float(kd)]["lean"])
        if best is None or loss < best[0]:
            best = (loss, float(kd))
    kd = best[1]
    return kd, fit_offset(dep, b, kd)


def shifted_axis(b: dict, kd: float, offset: np.ndarray) -> np.ndarray:
    return axis_of_lean(b["sides"][float(kd)]["lean"] + offset, b["yaw"])


def arm_stats(b: dict, dep: np.ndarray) -> dict:
    """All four arms' statistics for one depicted side of one clip."""
    k0 = float(K0)
    s = b["sides"][k0]
    ax0 = s["axis"]
    med0, win0 = tier_values(b, dep, ax0)
    off1 = fit_offset(dep, b, k0)
    med1, win1 = tier_values(b, dep, shifted_axis(b, k0, off1))
    k2, off2 = fit_kd(dep, b)
    med2, win2 = tier_values(b, dep, shifted_axis(b, k2, off2))
    coh = coherence_stat(b, dep, s["lean"])
    return {"F0_clip": med0, "F0_window": win0,
            "C1_clip": med1, "C1_window": win1,
            "C2_clip": med2, "C2_window": win2,
            "R_window": coh, "k_d_C2": k2}


# --------------------------------------------------------------------------- thresholds
def lofo_tau(values: np.ndarray, flights: np.ndarray, heldout: set[str],
             i: int | None = None, alpha: float = 0.05) -> float | np.ndarray:
    def one(j):
        m = (flights != flights[j]) & np.array([f not in heldout for f in flights])
        fit = values[m]
        fit = fit[np.isfinite(fit)]
        return float(np.quantile(fit, 1.0 - alpha)) if len(fit) >= 5 else np.nan

    if i is not None:
        return one(i)
    return np.array([one(j) for j in range(len(values))])


def mdv_of(curve: dict, thresh: float = 0.5) -> float:
    ks = sorted(curve)
    for i, k in enumerate(ks):
        if curve[k] >= thresh:
            if i == 0:
                return float(k)
            a, b = ks[i - 1], k
            ra, rb = curve[a], curve[b]
            return float(k if rb <= ra else a + (thresh - ra) * (b - a) / (rb - ra))
    return float("nan")


# =========================================================================== PART A
def part_a(data: dict, keep: dict) -> dict:
    rows = []
    for (cid, cond), b in sorted(data.items()):
        if cond == "real":
            st = arm_stats(b, b["dep"])
            rows.append({"clip_id": cid, "flight": b["flight"], "cond": "real",
                         "coverage": "full", **st})
        elif cond in FAMILIES or cond in CONTROLS:
            real = data[(cid, "real")]
            for cov, frac in COVERAGES:
                dep = b["dep"] if frac >= 1.0 else compose_partial(real["dep"], b["dep"], frac)
                st = arm_stats(b, dep)
                rows.append({"clip_id": cid, "flight": b["flight"], "cond": cond,
                             "coverage": cov, **st})
    rows = [r for r in rows if keep.get(r["clip_id"], False)]

    def table(arm: str, tier: str) -> dict:
        real = [r for r in rows if r["cond"] == "real" and r["flight"] in DEV]
        fl = np.array([r["flight"] for r in real])
        vals = np.array([r[f"{arm}_{tier}"] for r in real], float)
        taus = lofo_tau(vals, fl, HELDOUT)
        fr = int(np.nansum(vals > taus))
        entry = {"false_reject": f"{fr}/{len(vals)}", "tau_lofo_median": float(np.nanmedian(taus))}
        for fam in FAMILIES + CONTROLS:
            for cov, _ in COVERAGES:
                sub = [r for r in rows if r["flight"] in DEV and r["cond"] == fam
                       and r["coverage"] == cov]
                hits = tot = 0
                for r in sub:
                    m = fl == r["flight"]
                    if not m.any():
                        continue
                    hits += float(r[f"{arm}_{tier}"] > taus[np.flatnonzero(m)[0]])
                    tot += 1
                entry[f"{fam}/{cov}"] = f"{int(hits)}/{tot}"
        return entry

    tabs = {f"{arm}/{tier}": table(arm, tier)
            for arm in ("F0", "C1", "C2", "R") for tier in ("clip", "window")
            if not (arm == "R" and tier == "clip")}

    # matched false-reject: place each arm's cut at F0's false-reject count, pooled
    matched = {}
    for tier in ("clip", "window"):
        real = [r for r in rows if r["cond"] == "real" and r["flight"] in DEV]
        vals0 = np.array([r[f"F0_{tier}"] for r in real], float)
        fl0 = np.array([r["flight"] for r in real])
        tau0 = lofo_tau(vals0, fl0, HELDOUT)
        budget = int(np.nansum(vals0 > tau0))
        base = np.sort(vals0[np.isfinite(vals0)])
        cut = float(np.nextafter(base[-budget], -np.inf)) if budget else float(base.max() + 1)
        for arm in ("F0", "C1", "C2", "R"):
            if arm == "R" and tier == "clip":
                continue
            key = f"{'R' if arm == 'R' else arm}_{tier}"
            subreal = np.array([r[key] for r in real], float)
            subreal = subreal[np.isfinite(subreal)]
            # each statistic's own empirical cut at F0's false-reject count
            armcut = (float(np.nextafter(np.sort(subreal)[-budget], -np.inf))
                      if budget and len(subreal) >= budget else float(subreal.max() + 1))
            entry = {"threshold": armcut,
                     "false_reject": f"{int(np.sum(subreal > armcut))}/{len(subreal)}"}
            for fam in FAMILIES + CONTROLS:
                for cov, _ in COVERAGES:
                    v = np.array([r[key] for r in rows if r["flight"] in DEV
                                  and r["cond"] == fam and r["coverage"] == cov], float)
                    v = v[np.isfinite(v)]
                    entry[f"{fam}/{cov}"] = f"{int(np.sum(v > armcut))}/{len(v)}"
            matched[f"{arm}/{tier}"] = entry

    # descriptive flight-clustered intervals for the natural full-coverage rates
    def cluster_ci(arm: str, tier: str, fam: str) -> list[float]:
        real = [r for r in rows if r["cond"] == "real" and r["flight"] in DEV]
        fl = np.array([r["flight"] for r in real])
        vals = np.array([r[f"{arm}_{tier}"] for r in real], float)
        taus = lofo_tau(vals, fl, HELDOUT)
        sub = [r for r in rows if r["flight"] in DEV and r["cond"] == fam
               and r["coverage"] == "full"]
        per = {}
        for f in sorted(DEV):
            v = np.array([r[f"{arm}_{tier}"] for r in sub if r["flight"] == f], float)
            v = v[np.isfinite(v)]
            m = np.flatnonzero(fl == f)
            thr = taus[m[0]] if len(m) else np.nan
            per[f] = (int(np.sum(v > thr)), int(len(v)))
        rng = np.random.default_rng(1)
        keys = sorted(per)
        boots = []
        for _ in range(5000):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            h = n = 0
            for j in pick:
                a, b = per[keys[j]]
                h += a
                n += b
            boots.append(h / n if n else np.nan)
        return [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]

    intervals = {}
    for arm in ("F0", "C1", "C2", "R"):
        for tier in ("clip", "window"):
            if arm == "R" and tier == "clip":
                continue
            for fam in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
                        "yaw_control", "null_edit"):
                intervals[f"{arm}/{tier}/{fam}"] = cluster_ci(arm, tier, fam)

    # LOFO-matched operating point: the same leave-one-flight-out procedure at an alpha
    # chosen per arm so that its false-reject equals F0's natural count. This is the
    # like-for-like comparison; the pooled empirical cut above is also reported.
    matched_lofo = {}
    for tier in ("clip", "window"):
        real = [r for r in rows if r["cond"] == "real" and r["flight"] in DEV]
        fl = np.array([r["flight"] for r in real])
        basev = np.array([r[f"F0_{tier}"] for r in real], float)
        budget = int(np.nansum(basev > lofo_tau(basev, fl, HELDOUT)))
        for arm in ("F0", "C1", "C2", "R"):
            if arm == "R" and tier == "clip":
                continue
            key = f"{'R' if arm == 'R' else arm}_{tier}"
            vals = np.array([r[key] for r in real], float)
            best = None
            for alpha in np.arange(0.02, 0.121, 0.002):
                taus = lofo_tau(vals, fl, HELDOUT, alpha=alpha)
                fr = int(np.nansum(vals > taus))
                score = (abs(fr - budget), -alpha)
                if best is None or score < best[0]:
                    best = (score, float(alpha), taus)
            _, alpha, taus = best
            entry = {"alpha": alpha, "false_reject": f"{int(np.nansum(vals > taus))}/{len(vals)}",
                     "tau_lofo_median": float(np.nanmedian(taus))}
            for fam in FAMILIES + CONTROLS:
                for cov, _ in COVERAGES:
                    sub = [r for r in rows if r["flight"] in DEV and r["cond"] == fam
                           and r["coverage"] == cov]
                    hits = tot = 0
                    for r in sub:
                        m = fl == r["flight"]
                        if not m.any():
                            continue
                        hits += float(r[key] > taus[np.flatnonzero(m)[0]])
                        tot += 1
                    entry[f"{fam}/{cov}"] = f"{int(hits)}/{tot}"
            matched_lofo[f"{arm}/{tier}"] = entry

    absorption = {}
    for tier in ("clip", "window"):
        a = {}
        for fam in FAMILIES:
            for cov, _ in COVERAGES:
                def num(arm):
                    k, n = tabs[f"{arm}/{tier}"][f"{fam}/{cov}"].split("/")
                    return int(k), int(n)
                f0, c2 = num("F0"), num("C2")
                a[f"{fam}/{cov}"] = {
                    "F0": f0, "C2": c2,
                    "absorption_hits": f0[0] - c2[0],
                    "absorption_pp": (100.0 * f0[0] / f0[1] - 100.0 * c2[0] / c2[1])
                    if f0[1] and c2[1] else float("nan")}
        absorption[tier] = a

    # local-step MDV, J03's injection on the exact k_d = 0.295 lean
    k0 = float(K0)
    mdv_rows = []
    for (cid, cond), b in sorted(data.items()):
        if cond != "real" or not keep.get(cid, False):
            continue
        req = b["sides"][k0]["lean"]
        mag = np.linalg.norm(req, axis=-1, keepdims=True)
        lhat = np.divide(req, mag, out=np.zeros_like(req), where=mag > 1e-9)
        for frac, cov in ((0.25, "partial_25"), (0.50, "partial_50")):
            for delta in DELTAS:
                target = b["dep"] - np.sin(np.deg2rad(delta)) * lhat
                m = np.linalg.norm(target, axis=-1, keepdims=True)
                target = np.where(m > 0.995, target * (0.995 / np.maximum(m, 1e-12)), target)
                dep = compose_partial(b["dep"], target, frac)
                st = arm_stats(b, dep)
                mdv_rows.append({"clip_id": cid, "flight": b["flight"], "coverage": cov,
                                 "delta": float(delta), **st})
    keys = ("F0_clip", "C1_clip", "C2_clip", "F0_window", "C1_window", "C2_window", "R_window")
    curves = {}
    for key in keys:
        entry = {}
        for cov in ("partial_25", "partial_50"):
            curve = {}
            for delta in DELTAS:
                hits = tot = 0
                base = np.array([x[key] for x in mdv_rows
                                 if x["delta"] == 0.0 and x["coverage"] == cov
                                 and x["flight"] in DEV], float)
                base_fl = np.array([x["flight"] for x in mdv_rows
                                    if x["delta"] == 0.0 and x["coverage"] == cov
                                    and x["flight"] in DEV])
                for r in mdv_rows:
                    if r["flight"] not in DEV or r["delta"] != delta or r["coverage"] != cov:
                        continue
                    j = np.flatnonzero(base_fl == r["flight"])
                    if not len(j):
                        continue
                    fit = np.delete(base, j[0])
                    fit = fit[np.isfinite(fit)]
                    if len(fit) < 5:
                        continue
                    hits += float(r[key] > float(np.quantile(fit, 0.95)))
                    tot += 1
                curve[float(delta)] = round(hits / tot, 4) if tot else float("nan")
            entry[cov] = {"detection": curve, "mdv_deg": mdv_of(curve)}
        curves[key] = entry

    kds = [r["k_d_C2"] for r in rows if r["cond"] == "real"]
    diag = {"k_d_hist_real_C2": {str(k): int(np.sum(np.isclose(kds, k))) for k in GRID},
            "k_d_median_real_C2": float(np.median(kds)) if kds else float("nan")}
    return {"table": tabs, "matched": matched, "matched_lofo": matched_lofo,
            "intervals": intervals,
            "absorption": absorption,
            "mdv": curves, "diagnostics": diag, "rows": rows,
            "mdv_rows": mdv_rows}


# =========================================================================== PART B
MAXN = None


def apply_violation(req: np.ndarray, delta_deg: float) -> np.ndarray:
    mag = np.linalg.norm(req, axis=-1, keepdims=True)
    lhat = np.divide(req, mag, out=np.zeros_like(req), where=mag > 1e-9)
    new = np.clip(mag - np.sin(np.deg2rad(delta_deg)), 0.0, None)
    return lhat * new


def axis_angle_deg_lean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n2a = np.clip(np.sum(a ** 2, axis=-1), 0.0, 1.0)
    n2b = np.clip(np.sum(b ** 2, axis=-1), 0.0, 1.0)
    cos = np.sum(a * b, axis=-1) + np.sqrt((1.0 - n2a) * (1.0 - n2b))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def windows_of(resid: np.ndarray, length: int, hop: int) -> np.ndarray:
    n_clip, n = resid.shape
    starts = list(range(0, n - length + 1, hop))
    if not starts:
        return np.full((n_clip, 1), np.nan)
    view = np.stack([resid[:, s:s + length] for s in starts], axis=1)
    ok = np.sum(np.isfinite(view), axis=2) >= 6
    with np.errstate(all="ignore"):
        med = np.nanmedian(view, axis=2)
    med[~ok] = np.nan
    return med


def lofo_tau_arr(stats: np.ndarray, flights: np.ndarray) -> np.ndarray:
    out = np.full(len(stats), np.nan)
    for f in np.unique(flights):
        fit = stats[(flights != f) & np.isfinite(stats)]
        if len(fit) >= 5:
            out[flights == f] = float(np.quantile(fit, 0.95))
    return out


def run_cell(units: list[dict], flights: np.ndarray, req: np.ndarray, *, draws: int,
             windows, rng, mode: str, error_fn=None) -> dict:
    n_clip = len(units)
    T = {L: np.full((draws, len(DELTAS), n_clip), np.nan) for L in windows}
    for d in range(draws):
        err = np.zeros((n_clip, MAXN, 2))
        for f in np.unique(flights):
            ii = np.flatnonzero(flights == f)
            dt = float(np.median(np.diff(units[ii[0]]["t"])))
            if error_fn is not None:
                err[ii] = error_fn(units, ii, dt, rng)
                continue
            ang = rng.uniform(0.0, 2.0 * np.pi)
            bv = np.deg2rad(BIAS * np.array([np.cos(ang), np.sin(ang)]))
            for j in ii:
                e = make_error(MAXN, dt, SIGMA, TAU_S, FAST_SHARE, rng)
                err[j] = e + bv[None, :]
        for di, delta in enumerate(DELTAS):
            dep = apply_violation(req, float(delta)) + err
            resid = axis_angle_deg_lean(dep, req)
            for L in windows:
                if L is None:
                    with np.errstate(all="ignore"):
                        T[L][d, di, :] = np.nanmedian(resid, axis=1)
                else:
                    T[L][d, di, :] = np.nanmax(windows_of(resid, L, max(1, L // 2)), axis=1)
    taus = {L: np.stack([lofo_tau_arr(T[L][dr, 0, :], flights)
                         for dr in range(draws)]) for L in windows}
    out = {}
    for L in windows:
        det = {float(delta): float(np.nanmean(T[L][:, i, :] > taus[L]))
               for i, delta in enumerate(DELTAS)}
        out[str(L)] = {"h0_median_deg": float(np.nanmedian(T[L][:, 0, :])),
                       "tau_lofo_median_deg": float(np.nanmedian(taus[L])),
                       "detection": det, "mdv_deg": mdv_of(det),
                       "mode": mode}
    return out


def part_b(data: dict) -> dict:
    real = [data[(cid, "real")] for cid in sorted({c for c, cond in data if cond == "real"})]
    units = [{"t": b["t"], "flight": b["flight"], "n": len(b["t"])} for b in real]
    flights = np.array([u["flight"] for u in units])
    global MAXN
    MAXN = max(u["n"] for u in units)
    reqs = {}
    masked = {}
    for mode, kd in (("kd_0.295", float(K0)), ("drag_free", 0.0)):
        arr = np.full((len(real), MAXN, 2), np.nan)
        for j, b in enumerate(real):
            s = b["sides"][kd]
            arr[j, :len(b["t"])] = np.where(s["upright"][:, None], s["lean"], np.nan)
        reqs[mode] = arr
        masked[mode] = int(np.sum(~np.isfinite(arr[:, :, 0])) - sum(MAXN - u["n"] for u in units))
    rng = np.random.default_rng(1)
    out = {"masked_frames": masked, "windows": {}}
    for mode in ("kd_0.295", "drag_free"):
        out["windows"][mode] = run_cell(units, flights, reqs[mode], draws=J02_DRAWS,
                                        windows=[None, 24, 16, 12, 8], rng=rng, mode=mode)
    # parameter grid at the whole clip
    grid = {}
    for mode in ("kd_0.295", "drag_free"):
        for sigma in (3.0, 4.8, 8.0):
            for bias in (0.0, 2.0, 5.0):
                global SIGMA, BIAS
                old = (SIGMA, BIAS)
                SIGMA, BIAS = sigma, bias
                r = run_cell(units, flights, reqs[mode], draws=GRID_DRAWS, windows=[None],
                             rng=rng, mode=mode)
                SIGMA, BIAS = old
                grid[f"{mode}/sigma{sigma}/bias{bias}"] = r["None"]
    out["grid"] = grid
    # measured reader error replay: dep = exact required + (depicted - truth attitude)
    errs = []
    for b in real:
        e = b["dep"] - b["lean_act"]
        e[~b["sides"][float(K0)]["upright"]] = np.nan
        errs.append(e)

    def replay(units_, ii, dt_, rng_):
        out_ = np.zeros((len(ii), MAXN, 2))
        for k, j in enumerate(ii):
            out_[k, : units_[j]["n"]] = errs[j]
        return out_

    out["replay"] = {}
    for mode in ("kd_0.295", "drag_free"):
        out["replay"][mode] = run_cell(units, flights, reqs[mode], draws=1,
                                       windows=[None, 24, 16, 12, 8], rng=rng,
                                       mode=mode, error_fn=replay)
    # mocap-attitude anchor
    out["anchor"] = {}
    for mode in ("kd_0.295", "drag_free"):
        base = np.full((len(real), MAXN, 2), np.nan)
        for j, b in enumerate(real):
            base[j, :len(b["t"])] = np.where(b["sides"][float(K0)]["upright"][:, None],
                                             b["lean_act"], np.nan)
        hs, ts = [], []
        for _ in range(ANCHOR_DRAWS):
            err = np.zeros_like(base)
            for f in np.unique(flights):
                ii = np.flatnonzero(flights == f)
                dt = float(np.median(np.diff(units[ii[0]]["t"])))
                ang = rng.uniform(0, 2 * np.pi)
                bv = np.deg2rad(BIAS * np.array([np.cos(ang), np.sin(ang)]))
                for j in ii:
                    e = make_error(MAXN, dt, SIGMA, TAU_S, FAST_SHARE, rng)
                    err[j] = e + bv[None, :]
            resid = axis_angle_deg_lean(base + err, reqs[mode])
            with np.errstate(all="ignore"):
                vals = np.nanmedian(resid, axis=1)
            hs.append(float(np.nanmedian(vals)))
            tau = lofo_tau_arr(vals, flights)
            ts.append(float(np.nanmedian(tau)))
        out["anchor"][mode] = {"h0_median_deg": float(np.median(hs)),
                               "threshold_lofo_median_deg": float(np.median(ts)),
                               "h0_p10_p90_deg": [float(np.percentile(hs, 10)),
                                                  float(np.percentile(hs, 90))],
                               "draws": ANCHOR_DRAWS}
    return out


# =========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["A", "B", "all"])
    ap.add_argument("--linear", action="store_true",
                    help="use J03's linear construction (validation only, not the K03 result)")
    args = ap.parse_args()
    import datetime
    run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    keep = gate_keep()
    data = load_clips()
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    if args.linear:
        build_sides_linear(data)
    else:
        build_sides(data, kin)
    doc = {"run": run, "grid": GRID.tolist(), "huber_deg": HUBER_DEG,
           "construction": "linear_J03_validation" if args.linear else "exact",
           "population": {"clips": len({c for c, _ in data}), "gate_kept": sum(keep.values()),
                          "dev": sorted(DEV), "heldout": sorted(HELDOUT)}}
    if args.part in ("A", "all"):
        print("Part A ...", flush=True)
        doc["A"] = part_a(data, keep)
        _dump_a(doc["A"], out)
    if args.part in ("B", "all"):
        print("Part B ...", flush=True)
        doc["B"] = part_b(data)
        _dump_b(doc["B"], out)
    (out / "k03.json").write_text(json.dumps(_jsonable(doc), indent=1))
    _summary(doc)
    print("wrote", out)
    return 0


def _dump_a(A: dict, out: Path) -> None:
    (out / "A_table.json").write_text(json.dumps(_jsonable(A["table"]), indent=1))
    (out / "A_matched.json").write_text(json.dumps(_jsonable(A["matched"]), indent=1))
    (out / "A_matched_lofo.json").write_text(json.dumps(_jsonable(A["matched_lofo"]), indent=1))
    (out / "A_intervals.json").write_text(json.dumps(_jsonable(A["intervals"]), indent=1))
    (out / "A_absorption.json").write_text(json.dumps(_jsonable(A["absorption"]), indent=1))
    (out / "A_mdv.json").write_text(json.dumps(_jsonable(A["mdv"]), indent=1))
    (out / "A_diagnostics.json").write_text(json.dumps(_jsonable(A["diagnostics"]), indent=1))
    keys = ("F0_clip", "C1_clip", "C2_clip", "F0_window", "C1_window", "C2_window",
            "R_window", "k_d_C2")
    with open(out / "A_rows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "flight", "cond", "coverage", *keys])
        for r in A["rows"]:
            w.writerow([r["clip_id"], r["flight"], r["cond"], r["coverage"]]
                       + [r.get(k, "") for k in keys])
    with open(out / "A_mdv_rows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "flight", "coverage", "delta", *keys[:-1]])
        for r in A["mdv_rows"]:
            w.writerow([r["clip_id"], r["flight"], r["coverage"], r["delta"]]
                       + [r.get(k, "") for k in keys[:-1]])


def _dump_b(B: dict, out: Path) -> None:
    (out / "B_floors.json").write_text(json.dumps(_jsonable(B), indent=1))


def _summary(doc: dict) -> None:
    if "A" in doc:
        print("A table:")
        for k, v in doc["A"]["table"].items():
            print(" ", k, v["false_reject"],
                  {fam: v.get(fam + "/full") for fam in
                   ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
                    "yaw_control", "null_edit")})
        print("A matched LOFO:", {k: (v["alpha"], v["false_reject"],
                                       v.get("tilt_removal/full"), v.get("mirror_roll/full"),
                                       v.get("scale_1.5/full"))
                                   for k, v in doc["A"]["matched_lofo"].items()})
        print("A absorption (clip):", {k: (v["absorption_hits"], round(v["absorption_pp"], 1))
                                       for k, v in doc["A"]["absorption"]["clip"].items()
                                       if k.endswith("/full")})
        print("A MDV:", {k: {c: round(v[c]["mdv_deg"], 2) for c in v}
                         for k, v in doc["A"]["mdv"].items()})
        print("A k_d:", doc["A"]["diagnostics"])
    if "B" in doc:
        print("B floors:")
        for mode, wins in doc["B"]["windows"].items():
            print(" ", mode, {L: (round(v["h0_median_deg"], 2),
                                   round(v["tau_lofo_median_deg"], 2),
                                   round(v["mdv_deg"], 2)) for L, v in wins.items()})
        print("B replay:", {m: {L: round(v["mdv_deg"], 2) for L, v in r.items()}
                            for m, r in doc["B"]["replay"].items()})
        print("B anchor:", doc["B"]["anchor"])


if __name__ == "__main__":
    raise SystemExit(main())
