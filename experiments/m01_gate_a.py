"""Track M, Stage 1: Gate A (path-side signal strength), development analysis.

Pre-registered in ``configs/m01_gate_a_prereg.yaml``.  CPU only, zero GPU spend.

What it does
------------
1. Reproduces Track J's coherence arm (``results/j03_perclip/rows.csv``, schema-v2 required
   side) as a machinery check before anything else.
2. Builds the **schema-v3 exact required lean** for every TII clip: the unit thrust axis
   ``unit(a + g e3 + k_d v)`` at the native 274.73 Hz mocap rate, sampled at the clip's frame
   times, ``k_d = 0.295`` (airframe model M0 imported byte-identically from
   ``track-k-calibration``).  The clip-median residual on this side reproduces Track K's
   published exact threshold 8.87 deg.
3. Computes the path-side signal statistics and the coherence tier per condition.
4. Measures the reader noise ``sigma_n`` on development flights (the SNR argument's input).
5. Runs the selective-prediction analysis for Gate A: nested leave-one-flight-out selection
   of the sigma threshold by conformal risk control on both risks (coherence false-reject and
   yaw-control flag rate), risk-coverage curves, reason codes, and detection tables.

Usage
-----
    PYTHONPATH=. pixi run python experiments/m01_gate_a.py
    PYTHONPATH=. pixi run python experiments/m01_gate_a.py --verify-only
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
from src.airframe.schema_v3 import required_lean_at
from src.verdict.gate_a import (NEAR_VERTICAL_DEG, implied_snr, signal_stats,
                                window_signal)
from src.verdict.schema import load_clip
from src.verdict.selective import (cluster_interval, conformal_quantile,
                                   crc_from_counts, rate_interval)

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/track_d/clips"
OUT = ROOT / "results/m01"
GATE_TABLE = ROOT / "results/a2_gate/20261109T201449Z_298ff20/clip_table.csv"
GATE_CONFIG = ROOT / "configs/gate_frozen.yaml"

DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]

WIN, HOP = 8, 4
VAR_FLOOR = 0.01
FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
DELAYS = ("delay_ms_100", "delay_ms_200", "delay_ms_300")
ALL_CONDS = ("real",) + FAMILIES + CONTROLS + DELAYS
ALPHA = 0.05
TAU_ALPHA = 0.05
LAM_GRID = np.round(np.arange(0.0, 16.01, 0.25), 2)


# --------------------------------------------------------------------------- inputs
def gate_keep() -> dict[str, bool]:
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    tau = float(g["tau"])
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= tau
            for r in csv.DictReader(open(GATE_TABLE))}


def v3_lean(kin, c) -> tuple[np.ndarray, np.ndarray]:
    req = required_lean_at(kin, "M0", {}, np.asarray(c.t, float), np.asarray(c.yaw, float))
    return req["lean"], req["tilt_deg"]


# --------------------------------------------------------------------------- tiers
def coherence_series(dep: np.ndarray, req: np.ndarray) -> float:
    """Max over windows of ``1 - min(r_x, r_y)`` -- Track J's R arm, verbatim rules."""
    vals = []
    n = len(dep)
    spans = [(0, n)] if n <= WIN else [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]
    for lo, hi in spans:
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < 6:
            continue
        rs = []
        for j in range(2):
            if np.std(d[:, j]) < VAR_FLOOR or np.std(q[:, j]) < VAR_FLOOR:
                rs = []
                break
            rs.append(float(np.corrcoef(d[:, j], q[:, j])[0, 1]))
        if rs:
            vals.append(1.0 - min(rs))
    return float(max(vals)) if vals else float("nan")


def coherence_windows(dep: np.ndarray, req: np.ndarray) -> list[float]:
    """Per-window ``1 - min(r_x, r_y)``; NaN where the window abstains."""
    n = len(dep)
    spans = [(0, n)] if n <= WIN else [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]
    out = []
    for lo, hi in spans:
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < 6:
            out.append(float("nan"))
            continue
        rs = []
        for j in range(2):
            if np.std(d[:, j]) < VAR_FLOOR or np.std(q[:, j]) < VAR_FLOOR:
                rs = []
                break
            rs.append(float(np.corrcoef(d[:, j], q[:, j])[0, 1]))
        out.append(1.0 - min(rs) if rs else float("nan"))
    return out


def coherence_whole(dep: np.ndarray, req: np.ndarray) -> dict:
    ok = np.isfinite(dep).all(1) & np.isfinite(req).all(1)
    d, q = dep[ok], req[ok]
    if len(d) < 3:
        return {"r_x": np.nan, "r_y": np.nan, "s_x": np.nan, "s_y": np.nan,
                "whole": np.nan, "sign": np.nan}
    rs, ss = [], []
    for j in range(2):
        if np.std(d[:, j]) < 1e-12 or np.std(q[:, j]) < 1e-12:
            rs.append(0.0)
            ss.append(0.5)
            continue
        rs.append(float(np.corrcoef(d[:, j], q[:, j])[0, 1]))
        a, b = d[:, j], q[:, j]
        keep = (np.abs(a) > 1e-4) & (np.abs(b) > 1e-4)
        ss.append(float(np.mean(np.sign(a[keep]) == np.sign(b[keep]))) if keep.any() else 0.5)
    return {"r_x": rs[0], "r_y": rs[1], "s_x": ss[0], "s_y": ss[1],
            "whole": 1.0 - min(rs), "sign": 1.0 - min(ss)}


# --------------------------------------------------------------------------- reproduction
def reproduce_j03() -> dict:
    rows = pd.read_csv(ROOT / "results/j03_perclip/rows.csv")
    R = rows[(rows.arm == "R") & (rows.coverage == "full")]
    real = R[(R.cond == "real") & (R.flight.isin(DEV))]
    out = {"n_dev": int(len(real)), "n_finite": int(real.window.notna().sum())}

    def lofo(cond):
        hit = tot = 0
        d = R[(R.cond == cond) & (R.flight.isin(DEV))]
        for _, row in d.iterrows():
            fit = real[real.flight != row.flight].window.values
            fit = fit[np.isfinite(fit)]
            if len(fit) < 5:
                continue
            hit += int(row.window > np.quantile(fit, 0.95))
            tot += 1
        return [int(hit), int(tot)]

    for cond in ("real",) + FAMILIES + CONTROLS:
        out[cond] = lofo(cond)
    out["expected"] = {"real": [3, 69], "tilt_removal": [54, 69], "mirror_roll": [55, 69],
                       "scale_0.5": [6, 69], "scale_1.5": [11, 69],
                       "yaw_control": [9, 69], "null_edit": [3, 69]}
    out["reproduces"] = all(out[k] == out["expected"][k] for k in out["expected"])
    return out


def verify_f0_threshold(coh: pd.DataFrame) -> dict:
    """Clip-median residual threshold on the v3 side must be Track K's published 8.87 deg."""
    from src.verdict.segments import frame_residual_deg
    keep = gate_keep()
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    rows = []
    for f in DEV:
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond != "real" or not keep.get(cid, False):
                continue
            c = load_clip(d / "verdict_clip.npz")
            lean3, _ = v3_lean(kin[f], c)
            resid = frame_residual_deg(np.asarray(c.depicted_lean, float), lean3,
                                       np.asarray(c.yaw, float))
            rows.append({"flight": f, "med": float(np.nanmedian(resid))})
    df = pd.DataFrame(rows)
    taus = []
    for _, r in df.iterrows():
        fit = df[df.flight != r.flight].med.values
        taus.append(float(np.quantile(fit, 0.95)))
    return {"lofo_median_tau_deg": float(np.median(taus)), "published": 8.87,
            "abs_diff": float(abs(np.median(taus) - 8.87)), "n_clips": int(len(df)),
            "passes": bool(abs(np.median(taus) - 8.87) < 0.01)}


# --------------------------------------------------------------------------- tables
def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = gate_keep()
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    sig_rows, coh_rows = [], []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in TII_FLIGHTS:
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond not in ALL_CONDS:
                continue
            c = load_clip(d / "verdict_clip.npz")
            if cid not in cache:
                cache[cid] = v3_lean(kin[f], c)
                lean3, tilt3 = cache[cid]
                st = signal_stats(lean3, tilt3)
                row = st.as_dict()
                ws = window_signal(lean3, tilt3, WIN, HOP)
                row.update({
                    "clip_id": cid, "flight": f,
                    "kept_S3": bool(keep.get(cid, False)), "in_dev": f in DEV,
                    "window_sigma_min_min": float(np.min([w["sigma_min"] for w in ws])),
                    "window_sigma_min_median": float(np.median([w["sigma_min"] for w in ws])),
                })
                sig_rows.append(row)
            lean3, _ = cache[cid]
            dep = np.asarray(c.depicted_lean, float)
            lean2 = np.asarray(c.lean_req, float) if c.lean_req is not None else None
            rec = {"clip_id": cid, "flight": f, "cond": cond,
                   "in_dev": f in DEV, "kept_S3": bool(keep.get(cid, False)),
                   "coh_v3": coherence_series(dep, lean3),
                   "coh_v2": coherence_series(dep, lean2) if lean2 is not None else np.nan,
                   "n_frames": int(len(dep)),
                   "wcoh_min_v3": float(np.nanmin(coherence_windows(dep, lean3)))
                   if np.isfinite(np.array(coherence_windows(dep, lean3))).any() else np.nan}
            rec.update({f"whole_v3_{k}": v for k, v in coherence_whole(dep, lean3).items()})
            coh_rows.append(rec)
    return pd.DataFrame(sig_rows), pd.DataFrame(coh_rows)


def noise_estimate() -> dict:
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    resid = []
    for f in DEV:
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond != "real":
                continue
            c = load_clip(d / "verdict_clip.npz")
            lean3, _ = v3_lean(kin[f], c)
            dep = np.asarray(c.depicted_lean, float)
            ok = np.isfinite(dep).all(1) & np.isfinite(lean3).all(1)
            if ok.sum() < 8:
                continue
            e = dep[ok] - lean3[ok]
            resid.append(e - np.median(e, axis=0))
    E = np.vstack(resid)
    mad = np.median(np.abs(E), axis=0) * 1.4826
    return {"per_axis_robust_deg": [float(np.degrees(x)) for x in mad],
            "per_axis_rms_deg": [float(np.degrees(x)) for x in E.std(axis=0)],
            "norm_median_deg": float(np.degrees(np.median(np.linalg.norm(E, axis=1)))),
            "sigma_n_deg": float(np.degrees(np.max(mad))),
            "n_frames": int(len(E)),
            "note": ("robust (MAD) per-axis std of depicted - required on the six "
                     "development flights' real clips; the larger axis is sigma_n")}


# --------------------------------------------------------------------------- gate evaluation
def _panel(sig: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    p = coh.merge(sig[["clip_id", "sigma_min", "sigma_p", "tilt_max", "tilt_med",
                       "frac_above_80", "window_sigma_min_min"]],
                  on="clip_id", how="left")
    return p


def accepted_mask(p: pd.DataFrame, lam: float, tilt_limit: float = NEAR_VERTICAL_DEG
                  ) -> pd.Series:
    return (p.sigma_min >= lam) & (p.tilt_max <= tilt_limit)


def tau_for(p: pd.DataFrame, cond: str, exclude_flight: str | None, lam: float,
            acc_mask: pd.Series) -> float:
    """Conformal threshold on the accepted real clips of the fit flights."""
    fit = p[(p.cond == cond) & acc_mask]
    if exclude_flight is not None:
        fit = fit[fit.flight != exclude_flight]
    vals = fit.coh_v3.values
    vals = vals[np.isfinite(vals)]
    # the published convention (G01, J03): the empirical 95th percentile, needing >= 5
    # fit clips.  conformal_quantile's finite-sample floor of 8 is too strict for a
    # leave-one-flight-out fold with two or three contributing flights.
    if len(vals) < 5:
        return float("nan")
    return float(np.quantile(vals, 1.0 - TAU_ALPHA))


def curve(p: pd.DataFrame, fit_flights: list[str], lam_grid=LAM_GRID) -> pd.DataFrame:
    """Risk-coverage on ``fit_flights`` with a LOFO-within-fit tau at every lam."""
    rows = []
    base = p[p.flight.isin(fit_flights) & p.kept_S3]
    n_clips_total = base.clip_id.nunique()
    for lam in lam_grid:
        acc = accepted_mask(base, lam)
        n_clips = base[acc].clip_id.nunique()
        res = {"lam": float(lam), "accepted": int(n_clips),
               "coverage": n_clips / max(n_clips_total, 1)}
        for cond in ("real", "yaw_control", "null_edit"):
            hit = 0
            sub = base[(base.cond == cond) & acc]
            for _, r in sub.iterrows():
                tau = tau_for(base, "real", r.flight, lam, acc)
                if np.isfinite(tau) and np.isfinite(r.coh_v3) and r.coh_v3 > tau:
                    hit += 1
            res[cond] = hit
        rows.append(res)
    return pd.DataFrame(rows)


def select_lam(curve_table: pd.DataFrame) -> dict:
    fr = crc_from_counts(curve_table.lam, curve_table.accepted, curve_table.real, ALPHA)
    yw = crc_from_counts(curve_table.lam, curve_table.accepted, curve_table.yaw_control,
                         ALPHA)
    lams = [x.lam for x in (fr, yw) if np.isfinite(x.lam)]
    joint = max(lams) if len(lams) == 2 else float("nan")
    return {"false_reject": fr.as_dict(), "yaw_control": yw.as_dict(),
            "lam_joint": float(joint)}


def nested_dev(p: pd.DataFrame) -> pd.DataFrame:
    """Honest dev estimate: for each flight, lam and tau fitted on the other five."""
    out = []
    for f in DEV:
        others = [x for x in DEV if x != f]
        tab = curve(p, others)
        sel = select_lam(tab)
        lam = sel["lam_joint"]
        if not np.isfinite(lam):
            out.append({"flight": f, "lam": np.nan, "selected": False,
                        "reason": "no threshold reaches the target on the other five"})
            continue
        sub = p[(p.flight == f) & p.kept_S3]
        acc = accepted_mask(sub, lam)
        rec = {"flight": f, "lam": float(lam), "selected": True,
               "accepted": int(sub[acc].clip_id.nunique()),
               "total": int(sub.clip_id.nunique())}
        fit_pool = p[(p.kept_S3) & (p.flight.isin(DEV))]
        fit_acc = accepted_mask(fit_pool, lam)
        tau = tau_for(fit_pool, "real", f, lam, fit_acc)
        for cond in ("real",) + FAMILIES + CONTROLS + DELAYS:
            rows = sub[(sub.cond == cond) & acc]
            hit = tot = abst = 0
            for _, r in rows.iterrows():
                tot += 1
                if not np.isfinite(r.coh_v3):
                    abst += 1
                elif np.isfinite(tau) and r.coh_v3 > tau:
                    hit += 1
            rec[f"{cond}_hit"] = hit
            rec[f"{cond}_n"] = tot
            rec[f"{cond}_abstain"] = abst
        out.append(rec)
    return pd.DataFrame(out)


def ungated(p: pd.DataFrame) -> dict:
    """The published tier with no Gate A (and with only the near-vertical rule)."""
    base = p[(p.in_dev.astype(bool)) & (p.kept_S3.astype(bool))]
    out = {}
    for label, mask in (("ungated", pd.Series(True, index=base.index)),
                        ("near_vertical_only", base.tilt_max <= NEAR_VERTICAL_DEG)):
        sub = base[mask]
        rec = {"accepted": int(sub.clip_id.nunique())}
        for cond in ("real",) + FAMILIES + CONTROLS + DELAYS:
            hit = abst = tot = 0
            for _, r in sub[sub.cond == cond].iterrows():
                tau = tau_for(sub, "real", r.flight, -1e9, mask)
                tot += 1
                if not np.isfinite(r.coh_v3):
                    abst += 1
                elif np.isfinite(tau) and r.coh_v3 > tau:
                    hit += 1
            rec[cond] = {"hit": hit, "n": tot, "abstain": abst}
        out[label] = rec
    return out


def window_gate(p: pd.DataFrame, lam: float) -> dict:
    """Secondary, window-level Gate A: windows below lam or near-vertical abstain."""
    base = p[(p.kept_S3.astype(bool)) & p.in_dev.astype(bool)]
    rec = {}
    for cond in ("real",) + FAMILIES + CONTROLS:
        rows = base[base.cond == cond]
        # clip statistic: max over *admissible* windows, with the clip-level rule as well
        sub = rows[accepted_mask(rows, lam)]
        hit = abst = tot = 0
        for _, r in sub.iterrows():
            tot += 1
            if not np.isfinite(r.coh_v3):
                abst += 1
        rec[cond] = {"n": tot, "abstain": abst}
    rec["note"] = ("window-level abstention is reported in the report from the per-window "
                   "tables; the clip-level rule is the primary pre-registered gate")
    return rec



def at_lam(p: pd.DataFrame, lam: float) -> dict:
    """Full dev read-out at one frozen lambda, with reason codes and clustered rates."""
    base = p[(p.in_dev.astype(bool)) & (p.kept_S3.astype(bool))]
    acc = accepted_mask(base, lam)
    out = {"lam": float(lam), "n_clips_total": int(base.clip_id.nunique()),
           "n_accepted": int(base[acc].clip_id.nunique())}
    out["coverage"] = out["n_accepted"] / max(out["n_clips_total"], 1)
    reasons = {"weak_signal": 0, "near_vertical": 0, "both": 0, "none": 0}
    for cid, g in base.groupby("clip_id"):
        weak = not np.isfinite(g.sigma_min.iloc[0]) or g.sigma_min.iloc[0] < lam
        vert = g.tilt_max.iloc[0] > NEAR_VERTICAL_DEG
        if weak and vert:
            reasons["both"] += 1
        elif weak:
            reasons["weak_signal"] += 1
        elif vert:
            reasons["near_vertical"] += 1
        else:
            reasons["none"] += 1
    out["reasons"] = reasons
    rates = {}
    for cond in ("real",) + FAMILIES + CONTROLS + DELAYS:
        sub = base[(base.cond == cond) & acc]
        hit = abst = 0
        for _, r in sub.iterrows():
            tau = tau_for(base, "real", r.flight, lam, acc)
            if not np.isfinite(r.coh_v3):
                abst += 1
            elif np.isfinite(tau) and r.coh_v3 > tau:
                hit += 1
        rates[cond] = {"hit": int(hit), "n": int(len(sub)), "abstain": int(abst)}
    out["rates"] = rates
    # clustered interval on the accepted real-clip false-reject and yaw rates
    flags, den = [], []
    for f in DEV:
        sub = base[(base.cond == "real") & acc & (base.flight == f)]
        tau = tau_for(base, "real", f, lam, acc)
        flags.append(float(np.mean([1.0 if (np.isfinite(r.coh_v3) and np.isfinite(tau)
                                           and r.coh_v3 > tau) else 0.0
                                    for _, r in sub.iterrows()])) if len(sub) else np.nan)
        den.append(len(sub))
    ok = np.isfinite(flags)
    out["real_rate_interval"] = {"per_flight": dict(zip(np.array(DEV)[ok],
                                                        np.array(flags)[ok].round(4).tolist())),
                                 **rate_interval(np.array(flags)[ok], np.array(DEV)[ok])}
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rep = reproduce_j03()
    (OUT / "j03_reproduction.json").write_text(json.dumps(rep, indent=1))
    print("J03 R-arm reproduction:", "OK" if rep["reproduces"] else "MISMATCH")
    if args.verify_only:
        return

    sig, coh = build_tables()
    sig.to_csv(OUT / "signal_by_clip.csv", index=False)
    coh.to_csv(OUT / "coherence_v3.csv", index=False)
    p = _panel(sig, coh)

    f0 = verify_f0_threshold(coh)
    (OUT / "f0_verification.json").write_text(json.dumps(f0, indent=1))
    print("F0 v3 threshold:", f0)
    if not f0["passes"]:
        raise SystemExit("v3 required-lean construction does not reproduce Track K's 8.87")

    noise = noise_estimate()
    (OUT / "noise.json").write_text(json.dumps(noise, indent=1))
    print("sigma_n:", round(noise["sigma_n_deg"], 3), "deg")

    # ---- risk-coverage and selection
    tab_all = curve(p, DEV)
    tab_all.to_csv(OUT / "risk_coverage_dev.csv", index=False)
    sel = select_lam(tab_all)
    noise["sigma_n_x_k"] = {f"k={k}": k * noise["sigma_n_deg"] for k in (1.5, 2.0, 3.0)}
    noise["lam_over_sigma_n"] = (sel["lam_joint"] / noise["sigma_n_deg"]
                                 if np.isfinite(sel["lam_joint"]) else float("nan"))
    (OUT / "noise.json").write_text(json.dumps(noise, indent=1))
    (OUT / "selection_dev.json").write_text(json.dumps(sel, indent=1))
    print("selection (in-sample dev):", json.dumps(sel, indent=1))

    nested = nested_dev(p)
    nested.to_csv(OUT / "nested_dev.csv", index=False)
    print(nested.to_string())

    un = ungated(p)
    (OUT / "ungated.json").write_text(json.dumps(un, indent=1))
    print("ungated:", json.dumps({k: {c: v for c, v in d.items() if c in
                                      ("accepted", "real", "yaw_control")}
                                  for k, d in un.items()}, indent=1))

    # ---- clustered intervals on the honest dev rates
    rows = []
    for f in DEV:
        r = nested[nested.flight == f]
        if r.empty or not bool(r.selected.iloc[0]):
            continue
        r = r.iloc[0]
        for cond in ("real", "yaw_control", "null_edit", "tilt_removal", "mirror_roll"):
            rows.append({"flight": f, "n": int(r[f"{cond}_n"]),
                         "hit": int(r[f"{cond}_hit"])})
    pooled = {}
    for cond in ("real", "yaw_control", "null_edit", "tilt_removal", "mirror_roll"):
        hs = nested[f"{cond}_hit"].values.astype(float)
        ns = nested[f"{cond}_n"].values.astype(float)
        ok = np.isfinite(hs)
        flags = np.where(ns[ok] > 0, hs[ok] / np.maximum(ns[ok], 1), 0.0)
        pooled[cond] = {"hits": int(np.nansum(hs)), "n": int(np.nansum(ns)),
                        **rate_interval(flags, np.array(DEV)[ok])}
    (OUT / "dev_rates_clustered.json").write_text(json.dumps(pooled, indent=1))
    print("pooled out-of-fold dev rates:", json.dumps(pooled, indent=1))

    # ---- read-out at the frozen SNR candidate lambda = 2 * sigma_n
    lam_frozen = round(2.0 * noise["sigma_n_deg"], 2)
    frz = at_lam(p, lam_frozen)
    frz["lam_rule"] = "2 x sigma_n (the middle pre-registered SNR candidate)"
    (OUT / "frozen_lambda_dev.json").write_text(json.dumps(frz, indent=1))
    print("at frozen lambda:", json.dumps(frz, indent=1))

    print("\nSignal by flight (kept, dev):")
    print(sig[sig.in_dev.astype(bool) & sig.kept_S3.astype(bool)]
          .groupby("flight")[["sigma_min", "sigma_p", "tilt_med", "tilt_max"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
