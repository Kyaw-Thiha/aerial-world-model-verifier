"""Track M, Stage 1: Gate A (path-side signal strength), development analysis.

Pre-registered in ``configs/m01_gate_a_prereg.yaml``.  CPU only, zero GPU spend.

What it does
------------
1. Builds the **schema-v3 exact required lean** for every TII clip: the unit thrust axis
   ``unit(a + g e3 + k_d v)`` at the native 274.73 Hz mocap rate, sampled at the clip's frame
   times, with ``k_d = 0.295`` (the schema's own model, imported byte-identically from
   ``track-k-calibration``).  This replaces schema v2's drag-free ``lean_req``; the F0
   threshold it reproduces (8.8707 deg) matches Track K's published 8.87 deg.
2. Reproduces Track J's coherence arm (``results/j03_perclip/rows.csv``, v2 required side) as
   a machinery check before using the v3 side.
3. Computes the coherence tier per condition on the v3 side and the path-side signal
   statistics per clip.
4. Measures the reader noise ``sigma_n`` the SNR argument needs, on development flights only.
5. Runs the selective-prediction analysis: risk-coverage curves and conformal risk control
   for Gate A, with both the clip tier and the window tier, including the LOFO-within-dev
   (honest) variant.

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
from src.verdict.gate_a import NEAR_VERTICAL_DEG, gate_a, signal_stats, window_signal
from src.verdict.schema import load_clip
from src.verdict.selective import (cluster_auroc, cluster_interval, conformal_quantile,
                                   crc_threshold, risk_coverage)

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


# --------------------------------------------------------------------------- inputs
def gate_keep() -> dict[str, bool]:
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    tau = float(g["tau"])
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= tau
            for r in csv.DictReader(open(GATE_TABLE))}


def v3_lean(kin, c) -> tuple[np.ndarray, np.ndarray]:
    req = required_lean_at(kin, "M0", {}, np.asarray(c.t, float),
                           np.asarray(c.yaw, float))
    return req["lean"], req["tilt_deg"]


# --------------------------------------------------------------------------- tier
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


def coherence_whole(dep: np.ndarray, req: np.ndarray) -> dict:
    """Whole-clip per-axis correlation and sign agreement (the frozen D/B tier)."""
    ok = np.isfinite(dep).all(1) & np.isfinite(req).all(1)
    d, q = dep[ok], req[ok]
    if len(d) < 3:
        return {"r_x": np.nan, "r_y": np.nan, "s_x": np.nan, "s_y": np.nan}
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
    return {"r_x": rs[0], "r_y": rs[1], "s_x": ss[0], "s_y": ss[1]}


# --------------------------------------------------------------------------- reproduction
def reproduce_j03() -> dict:
    """Recompute J03's R arm from its own stored rows (v2 required side)."""
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


# --------------------------------------------------------------------------- main
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
                st = signal_stats(cache[cid][0], cache[cid][1])
                row = st.as_dict()
                row.update({"clip_id": cid, "flight": f,
                            "kept_S3": bool(keep.get(cid, False)),
                            "in_dev": f in DEV})
                row["window_sigma_min_min"] = float(np.min(
                    [w["sigma_min"] for w in window_signal(cache[cid][0], cache[cid][1],
                                                           WIN, HOP)]))
                row["window_sigma_min_median"] = float(np.median(
                    [w["sigma_min"] for w in window_signal(cache[cid][0], cache[cid][1],
                                                           WIN, HOP)]))
                sig_rows.append(row)
            lean3, tilt3 = cache[cid]
            dep = np.asarray(c.depicted_lean, float)
            lean2 = np.asarray(c.lean_req, float) if c.lean_req is not None else None
            rec = {"clip_id": cid, "flight": f, "cond": cond,
                   "in_dev": f in DEV, "kept_S3": bool(keep.get(cid, False)),
                   "coh_v3": coherence_series(dep, lean3),
                   "coh_v2": coherence_series(dep, lean2) if lean2 is not None else np.nan,
                   "n_frames": int(len(dep))}
            rec.update({f"whole_v3_{k}": v for k, v in coherence_whole(dep, lean3).items()})
            coh_rows.append(rec)
    return pd.DataFrame(sig_rows), pd.DataFrame(coh_rows)


def noise_estimate(coh: pd.DataFrame) -> dict:
    """Per-axis robust reader noise on the development real clips (median absolute deviation)."""
    keep = gate_keep()
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    resid = []
    for f in TII_FLIGHTS:
        if f not in DEV:
            continue
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
                     "development flights' real clips; the larger axis is taken as "
                     "sigma_n for the SNR argument")}


def evaluate(panel: pd.DataFrame, tau_grid: np.ndarray) -> dict:
    """Risk-coverage and detection counts for Gate A, LOFO within dev and in-sample."""
    dev = panel[panel.in_dev & panel.kept_S3]
    out = {}
    # ---- risk-coverage (in-sample and LOFO-within-dev)
    for label, lofo in (("in_sample", False), ("lofo_dev", True)):
        rows = []
        for lam in tau_grid:
            cov = fr = yw = nl = 0
            n_clips = 0
            for cid, grp in dev.groupby("clip_id"):
                g = grp.set_index("cond")
                sig = float(g.sigma_min.iloc[0])
                ok = np.isfinite(sig) and sig >= lam and float(g.tilt_max.iloc[0]) <= NEAR_VERTICAL_DEG
                if not ok:
                    continue
                n_clips += 1
                fl = []
                for cond in ("real", "yaw_control", "null_edit"):
                    v = float(g.coh_v3.get(cond, np.nan)) if cond in g.index else np.nan
                    fit = dev[(dev.flight != g.flight.iloc[0]) & (dev.cond == "real")
                              & (dev.sigma_min >= lam)
                              & (dev.tilt_max <= NEAR_VERTICAL_DEG)].coh_v3.values if lofo else \
                          dev[(dev.cond == "real") & (dev.sigma_min >= lam)
                              & (dev.tilt_max <= NEAR_VERTICAL_DEG)].coh_v3.values
                    fit = fit[np.isfinite(fit)]
                    tau = np.quantile(fit, 0.95) if len(fit) >= 5 else np.nan
                    fl.append(float(np.isfinite(tau) and np.isfinite(v) and v > tau))
                cov += 1
                fr += fl[0]
                yw += fl[1]
                nl += fl[2]
            rows.append({"lam": float(lam), "accepted": cov,
                         "coverage": cov / max(len(dev.clip_id.unique()), 1),
                         "false_reject": int(fr), "yaw_control": int(yw), "null": int(nl)})
        out[label] = rows
    return out


def detection_table(panel: pd.DataFrame, lam: float, lofo: bool = True) -> dict:
    dev = panel[panel.in_dev & panel.kept_S3]
    acc = dev[(dev.sigma_min >= lam) & (dev.tilt_max <= NEAR_VERTICAL_DEG)]
    real = acc[acc.cond == "real"]
    res = {}
    for cond in ("real",) + FAMILIES + CONTROLS + DELAYS:
        hit = tot = nan = 0
        for _, r in acc[acc.cond == cond].iterrows():
            fit = real[(real.flight != r.flight) if lofo else (real.cond == "real")]
            fit = fit.coh_v3.values
            fit = fit[np.isfinite(fit)]
            if len(fit) < 5:
                continue
            tau = np.quantile(fit, 0.95)
            tot += 1
            if not np.isfinite(r.coh_v3):
                nan += 1
                continue
            hit += int(r.coh_v3 > tau)
        res[cond] = {"hits": int(hit), "n": int(tot), "abstained": int(nan)}
    # all clips (no gate), for the coverage comparison
    res["_ungated"] = {}
    for cond in ("real",) + FAMILIES + CONTROLS:
        hit = tot = nan = 0
        rows = dev[dev.cond == cond]
        fit_all = dev[dev.cond == "real"]
        for _, r in rows.iterrows():
            hit += int(np.isfinite(r.coh_v3) and
                       r.coh_v3 > np.quantile(fit_all[fit_all.flight != r.flight].coh_v3
                                             .dropna().values, 0.95))
            tot += 1
            nan += int(not np.isfinite(r.coh_v3))
        res["_ungated"][cond] = {"hits": int(hit), "n": int(tot), "abstained": int(nan)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rep = reproduce_j03()
    (OUT / "j03_reproduction.json").write_text(json.dumps(rep, indent=1))
    print("J03 R-arm reproduction:", "OK" if rep["reproduces"] else "MISMATCH", rep)
    if args.verify_only:
        return

    sig, coh = build_tables()
    panel = coh.merge(sig, on=["clip_id", "flight", "in_dev", "kept_S3"], how="left")
    sig.to_csv(OUT / "signal_by_clip.csv", index=False)
    coh.to_csv(OUT / "coherence_v3.csv", index=False)

    noise = noise_estimate(coh)
    (OUT / "noise.json").write_text(json.dumps(noise, indent=1))
    print("sigma_n:", noise["sigma_n_deg"], "deg per axis")

    grid = np.round(np.arange(0.0, 16.01, 0.25), 2)
    rc = evaluate(panel, grid)
    pd.DataFrame(rc["in_sample"]).to_csv(OUT / "risk_coverage_in_sample.csv", index=False)
    pd.DataFrame(rc["lofo_dev"]).to_csv(OUT / "risk_coverage_lofo_dev.csv", index=False)

    # ---- conformal risk control on the honest (LOFO-within-dev) curve
    lofo = pd.DataFrame(rc["lofo_dev"])
    out = {}
    for key in ("false_reject", "yaw_control"):
        risk = lofo[key].values / np.maximum(lofo.accepted.values, 1)
        res = crc_threshold(lofo.lam.values, risk, alpha=ALPHA, min_accept=10)
        out[key] = res.as_dict()
    lam = max(out["false_reject"].get("lam", np.nan) if np.isfinite(out["false_reject"].get("lam", np.nan)) else 0.0,
              out["yaw_control"].get("lam", np.nan) if np.isfinite(out["yaw_control"].get("lam", np.nan)) else 0.0)
    out["lam_joint"] = float(lam) if lam > 0 else float("nan")
    out["grid_used"] = "in-sample accept counts differ per lam; CRC uses the LOFO curve"
    (OUT / "gate_a_selection.json").write_text(json.dumps(out, indent=1))
    print("CRC:", json.dumps(out, indent=1))

    det = detection_table(panel, lam if np.isfinite(lam) else 0.0)
    (OUT / "detection_dev.json").write_text(json.dumps(det, indent=1))
    print(json.dumps(det, indent=1))

    print("\nSignal by flight (kept, dev):")
    d = sig[sig.in_dev & sig.kept_S3]
    print(d.groupby("flight")[["sigma_min", "sigma_p", "tilt_med", "tilt_max"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
