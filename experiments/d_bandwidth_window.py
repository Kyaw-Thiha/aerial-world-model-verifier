"""Label-free required-lean window from the commanded path, applied to TII and UZH.

Pre-registration: ``configs/window_bandwidth_prereg.yaml`` (c = 0.5, window = c / f95,
8-sample floor, abstain if the floor is not met).  The rule reads only ``pos`` and ``t``;
it never sees condition labels, edits or verdict outcomes.

TII uses dense mocap (274.73 Hz) for the required lean; UZH schema-v1 clips use their own
rate (about 15 Hz), so the 8-sample floor decides testability.  Bands: TII LOFO; UZH with
the TII band unchanged and with UZH leave-one-sequence-out bands.

    PYTHONPATH=. pixi run python experiments/d_bandwidth_window.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

from src.datasets.tii import load_tii
from src.datasets.tii_video import flight_parts
from src.verdict import (FitOptions, coherence, fit_bands, fit_gain, gate_kept_from_conf,
                         load_clip)
from src.verdict.abstain import assess
from src.verdict.decide import JointBand, ScalarBand, flag
from src.verdict.dense import dense_required_lean
from src.verdict.lean import required_lean
from src.verdict.schema import Clip

ROOT = Path(__file__).resolve().parents[1]
TII_FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
               "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
               "flight-13p-trackRATM", "flight-16p-trackRATM"]
TII_TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
CONDITIONS = ["real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control", "null_edit"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]
C = 0.5
FLOOR = 8
SPECTRUM_WINDOW_S = 0.10
POLY = 3


def f95_from_accel(acc: np.ndarray, dt: float) -> float:
    acc = np.asarray(acc, dtype=np.float64)
    acc = acc[np.isfinite(acc).all(axis=1)]
    if len(acc) < 12:
        return float("nan")
    acc = acc - acc.mean(axis=0)
    freq = np.fft.rfftfreq(len(acc), dt)
    energy = np.sum(np.abs(np.fft.rfft(acc, axis=0)) ** 2, axis=1)
    energy[0] = 0.0
    if energy.sum() <= 0:
        return float("nan")
    cum = np.cumsum(energy) / energy.sum()
    return float(freq[min(int(np.searchsorted(cum, 0.95)), len(freq) - 1)])


def spectrum_accel(pos: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = int(round(SPECTRUM_WINDOW_S / dt))
    if w % 2 == 0:
        w += 1
    w = max(POLY + 2, w)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    acc = savgol_filter(pos, w, POLY, deriv=2, delta=dt, axis=0, mode="interp")
    return acc, dt


def window_decision(f95: float, rate: float, duration: float) -> dict:
    if not np.isfinite(f95) or f95 < 0.05:
        return {"window_s": float("nan"), "samples": float("nan"),
                "ok": False, "reason": "f95_undefined"}
    w = C / f95
    samples = w * rate
    if samples < FLOOR:
        return {"window_s": float(w), "samples": float(samples), "ok": False,
                "reason": f"window_samples {samples:.1f} < {FLOOR}"}
    if w > 0.5 * duration:
        return {"window_s": float(w), "samples": float(samples), "ok": False,
                "reason": f"window {w:.2f}s > half duration {0.5*duration:.2f}s"}
    return {"window_s": float(w), "samples": float(samples), "ok": True, "reason": ""}


def load_uzh(path: Path) -> Clip:
    with np.load(path, allow_pickle=False) as d:
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        n = len(d["t"])
        return Clip(t=np.asarray(d["t"]), pos=np.asarray(d["pos"]),
                    yaw=np.asarray(d["yaw"]),
                    depicted_lean=np.asarray(d["depicted_lean"]),
                    depicted_conf=np.asarray(d["depicted_conf"]), meta=meta)


def real_testable(keys_clips: dict, results: dict, abstain_cfg: dict, gate: dict,
                  window: dict) -> dict:
    """Pair-level testability; window is keyed like results."""
    out = {}
    for key, clip in keys_clips.items():
        if clip.condition != "real":
            continue
        r = results.get(key)
        if r is None:
            out[key] = {"testable": False, "reasons": ["no fit"]}
            continue
        gk = gate_kept_from_conf(clip.depicted_conf, gate)
        ab = assess(r, abstain_cfg, gate_kept=gk)
        w = window.get(key, {"ok": True, "reason": ""})
        reasons = list(ab.reasons)
        if not w["ok"]:
            reasons.append(w["reason"])
        out[key] = {"testable": not reasons, "reasons": reasons,
                    "gate_kept": gk, "window_s": w.get("window_s"),
                    "window_samples": w.get("samples")}
    return out


def evaluate(results: dict, testable: dict, rule: str, bands=None, folds=None,
             alpha=0.05) -> dict:
    """LOFO/LOSO rates; ``bands`` fixed or fitted per fold."""
    frr, det = [], {c: [] for c in EDITS}
    fold_rows = []
    fold_keys = folds or []
    for held in fold_keys:
        if bands is None:
            fit = [r for (f, cid, cond), r in results.items()
                   if cond == "real" and f != held
                   and testable.get((f, cid, "real"), {}).get("testable")]
            b = fit_bands(fit, rule, alpha)
        else:
            b = bands
        fr, de = [], {c: [] for c in EDITS}
        for (f, cid, cond), r in results.items():
            if f != held or not testable.get((f, cid, "real"), {}).get("testable"):
                continue
            if cond == "real":
                fr.append(bool(flag(r, rule, b)))
                frr.append(bool(flag(r, rule, b)))
            elif cond in EDITS:
                de[cond].append(bool(flag(r, rule, b)))
                det[cond].append(bool(flag(r, rule, b)))
        fold_rows.append({"held_out": held, "n_real": len(fr),
                          "frr": float(np.mean(fr)) if fr else float("nan"),
                          **{f"det_{c}": float(np.mean(de[c])) if de[c] else float("nan")
                             for c in EDITS}})
    return {"frr": float(np.mean(frr)) if frr else float("nan"), "n_real": len(frr),
            **{f"det_{c}": float(np.mean(det[c])) if det[c] else float("nan")
               for c in EDITS},
            **{f"n_{c}": len(det[c]) for c in EDITS},
            "by_fold": fold_rows}


def frozen_bands() -> object:
    cfg = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
    a = cfg["bands"]["axis"]
    return fit_bands  # placeholder, replaced below


def load_frozen_band():
    cfg = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
    a = cfg["bands"]["axis"]
    return JointBand(tuple(a["keys"]), {k: float(v) for k, v in a["center"].items()},
                     {k: float(v) for k, v in a["scale"].items()},
                     float(a["threshold"]), 0.05, int(a["n"]))


def window_summary(df: pd.DataFrame, group: str) -> dict:
    out = {}
    for key, g in df.groupby(group):
        out[str(key)] = {"n": int(len(g)), "median_s": float(g.window_s.median()),
                         "min_s": float(g.window_s.min()),
                         "max_s": float(g.window_s.max()),
                         "median_samples": float(g.samples.median())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/d_bandwidth")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    out_dir = ROOT / args.out / "d01"
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    band_prereg = yaml.safe_load((ROOT / "configs/window_bandwidth_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    coh_cfg = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
    tau_c = float(coh_cfg["thresholds"]["tau_correlation"])
    tau_s = float(coh_cfg["thresholds"]["tau_sign"])
    frozen_band = load_frozen_band()

    # ---------------------------------------------------------------- TII
    tii_clips, tii_results, tii_windows = {}, {}, {}
    tii_dense_acc = {}
    for flight in TII_FLIGHTS:
        control, _ = flight_parts(flight)
        mocap_path = ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv"
        mocap = load_tii(mocap_path)
        acc, dt_dense = spectrum_accel(mocap.pos, mocap.t)
        tii_dense_acc[flight] = (mocap, acc, dt_dense)
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        if clip.condition not in CONDITIONS:
            continue
        flight = clip.flight
        mocap, acc, dt_dense = tii_dense_acc[flight]
        i0 = np.searchsorted(mocap.t, clip.t[0])
        i1 = np.searchsorted(mocap.t, clip.t[-1])
        f95 = f95_from_accel(acc[i0:i1], dt_dense)
        dec = window_decision(f95, 1.0 / dt_dense, clip.t[-1] - clip.t[0])
        key = (flight, clip.clip_id, clip.condition)
        tii_clips[key] = clip
        tii_windows[key] = {**dec, "f95_hz": f95, "rate_hz": 1.0 / dt_dense}
        if not dec["ok"]:
            tii_results[key] = fit_gain(
                replace(clip, depicted_lean=np.full_like(clip.depicted_lean, np.nan)),
                FitOptions(bootstrap=False))
            continue
        dense = dense_required_lean(mocap_path := ROOT / (
            f"data/raw/tii/{flight_parts(flight)[0]}/mocap_{flight}.csv"),
            clip.t, window_s=dec["window_s"])
        cw = replace(clip, lean_req=dense.l_req0, vel_req=dense.vel_req,
                     tilt_req_deg=dense.tilt_deg)
        tii_results[key] = fit_gain(cw, FitOptions(bootstrap=True,
                                                   n_boot=args.n_boot))
    tii_test = real_testable(tii_clips, tii_results, prereg["abstention"], gate,
                             tii_windows)
    tii_windows_df = pd.DataFrame([
        {"flight": f, "clip_id": cid, "condition": cond,
         **{k: v for k, v in w.items()}}
        for (f, cid, cond), w in tii_windows.items()
        if cond == "real" and tii_test.get((f, cid, "real"), {}).get("window_s") is not None])
    tii_windows_df.to_csv(out_dir / "tii_windows.csv", index=False)
    tii_rates = evaluate(tii_results, tii_test, "R2_axis_gains", folds=TII_FLIGHTS)
    tii_n = sum(1 for v in tii_test.values() if v["testable"])
    tii_total = sum(1 for k, c in tii_clips.items() if c.condition == "real")
    tii_summary = {"n_real_total": tii_total, "n_real_testable": tii_n,
                   "abstention": 1.0 - tii_n / max(tii_total, 1),
                   "rates": tii_rates, "windows_by_flight": window_summary(
                       tii_windows_df, "flight") if len(tii_windows_df) else {}}

    # ---------------------------------------------------------------- UZH
    uz_dir = ROOT / "data/processed/track_e_uzh/clips"
    uzh_clips, uzh_results, uzh_windows = {}, {}, {}
    if uz_dir.exists():
        for path in sorted(uz_dir.rglob("*.npz")):
            clip = load_uzh(path)
            seq = str(clip.meta.get("sequence"))
            if "forward" not in seq:
                continue
            cond = str(clip.meta.get("condition"))
            if cond not in CONDITIONS:
                continue
            acc, dt = spectrum_accel(clip.pos, clip.t)
            f95 = f95_from_accel(acc, dt)
            dec = window_decision(f95, 1.0 / dt, clip.t[-1] - clip.t[0])
            key = (seq, clip.meta["clip_id"], cond)
            uzh_clips[key] = clip
            uzh_windows[key] = {**dec, "f95_hz": f95, "rate_hz": 1.0 / dt}
            if not dec["ok"]:
                uzh_results[key] = fit_gain(
                    replace(clip, depicted_lean=np.full_like(clip.depicted_lean, np.nan)),
                    FitOptions(bootstrap=False))
                continue
            try:
                uzh_results[key] = fit_gain(
                    clip, FitOptions(bootstrap=True, n_boot=args.n_boot,
                                     sg_window_s=dec["window_s"]))
            except ValueError:
                uzh_results[key] = fit_gain(
                    replace(clip, depicted_lean=np.full_like(clip.depicted_lean, np.nan)),
                    FitOptions(bootstrap=False))
    uzh_test = real_testable(uzh_clips, uzh_results, prereg["abstention"], gate,
                             uzh_windows)
    uzh_n = sum(1 for v in uzh_test.values() if v["testable"])
    uzh_windows_df = pd.DataFrame([
        {"sequence": f, "clip_id": cid, "condition": cond,
         **{k: v for k, v in w.items()}}
        for (f, cid, cond), w in uzh_windows.items()
        if cond == "real" and uzh_test.get((f, cid, "real"), {}).get("window_s") is not None])
    if len(uzh_windows_df):
        uzh_windows_df.to_csv(out_dir / "uzh_windows.csv", index=False)
    uzh_summary = {"n_real_total": sum(1 for k, c in uzh_clips.items()
                                       if c.condition == "real"),
                   "n_real_testable": uzh_n,
                   "abstention": 1.0 - uzh_n / max(sum(1 for k, c in uzh_clips.items()
                                                       if c.condition == "real"), 1),
                   "windows_by_sequence": window_summary(uzh_windows_df, "sequence")
                   if len(uzh_windows_df) else {},
                   "rates_tii_band": None, "rates_uzh_loso": None}
    if uzh_n:
        seqs = sorted({k[0] for k in uzh_test})
        uzh_summary["rates_tii_band"] = evaluate(
            uzh_results, uzh_test, "R2_axis_gains", bands=frozen_band, folds=seqs)
        uzh_summary["rates_uzh_loso"] = evaluate(
            uzh_results, uzh_test, "R2_axis_gains", folds=seqs)

    # ---------------------------------------------------------------- coherence (UZH)
    coh_rows = []
    for key, clip in uzh_clips.items():
        if clip.condition not in CONDITIONS or key[2] != "real":
            continue
        # coherence at Track E's working 0.10 s required lean; lag from a gain fit at 0.10
        try:
            req = required_lean(clip.pos, clip.yaw, clip.t, window_s=0.10)
            cw = replace(clip, lean_req=req.l_req0, vel_req=req.velocity_reg,
                         tilt_req_deg=req.tilt_deg)
            r = fit_gain(cw, FitOptions(bootstrap=False, sg_window_s=0.10))
            coh = coherence(cw, lag_ms=r.lag_ms)
            coh_rows.append({"sequence": key[0], "clip_id": key[1],
                             "r_min": coh.r_min, "s_min": coh.s_min,
                             "correlation_score": coh.correlation_score,
                             "sign_score": coh.sign_score})
        except Exception:
            continue
    coh = {}
    if coh_rows:
        cdf = pd.DataFrame(coh_rows)
        cdf.to_csv(out_dir / "uzh_coherence.csv", index=False)
        coh = {"n_real": int(len(cdf)),
               "FRR_corr": float((cdf.correlation_score > tau_c).mean()),
               "FRR_sign": float((cdf.sign_score > tau_s).mean()),
               "r_min_median": float(cdf.r_min.median()),
               "s_min_median": float(cdf.s_min.median())}

    # ---------------------------------------------------------------- band stability (TII)
    real_tii = [r for (f, cid, cond), r in tii_results.items()
                if cond == "real" and tii_test.get((f, cid, "real"), {}).get("testable")]
    rng = np.random.default_rng(20261114)
    stability = {}
    for n in (6, 10, 15, 20, 25, 30, 40, 50):
        if n > len(real_tii):
            continue
        gx, gy, tau = [], [], []
        for _ in range(300):
            idx = rng.choice(len(real_tii), size=n, replace=False)
            b = fit_bands([real_tii[i] for i in idx], "R2_axis_gains", 0.05)
            if b.axis:
                gx.append(b.axis.center["gx"]); gy.append(b.axis.center["gy"])
                tau.append(b.axis.threshold)
        if gx:
            stability[str(n)] = {
                "gx_iqr": float(np.percentile(gx, 75) - np.percentile(gx, 25)),
                "gy_iqr": float(np.percentile(gy, 75) - np.percentile(gy, 25)),
                "tau_iqr": float(np.percentile(tau, 75) - np.percentile(tau, 25)),
                "tau_rel_iqr": float((np.percentile(tau, 75)
                                      - np.percentile(tau, 25)) / np.median(tau)),
                "median_tau": float(np.median(tau)),
            }
    metrics = {"prereg": "configs/window_bandwidth_prereg.yaml",
               "c": C, "floor_samples": FLOOR,
               "tii": tii_summary, "uzh": uzh_summary,
               "uzh_coherence_0.10s": coh,
               "band_stability_tii": stability,
               "band_stability_min_n": next(
                   (int(n) for n, v in stability.items()
                    if v["tau_rel_iqr"] < 0.10), None)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))

    lines = ["# Label-free bandwidth window", "",
             f"c = {C}, window = c / f95, floor {FLOOR} samples", "",
             "## TII", "",
             f"testable real: {tii_n}/{tii_total}; "
             f"abstention {100*(1-tii_n/max(tii_total,1)):.1f}%", "",
             "| flight | n | median window s | min | max | samples |",
             "| --- | --- | --- | --- | --- | --- |"]
    for f, v in tii_summary["windows_by_flight"].items():
        lines.append(f"| {f} | {v['n']} | {v['median_s']:.3f} | {v['min_s']:.3f} | "
                     f"{v['max_s']:.3f} | {v['median_samples']:.0f} |")
    r = tii_rates
    lines += ["", f"R2: FRR {r['frr']:.3f} (n={r['n_real']}), removal {r['det_tilt_removal']:.3f}, "
              f"mirror {r['det_mirror_roll']:.3f}, x0.5 {r['det_scale_0.5']:.3f}, "
              f"x1.5 {r['det_scale_1.5']:.3f}, yaw {r['det_yaw_control']:.3f}", ""]
    lines += ["## UZH forward", "",
              f"real testable: {uzh_n}/{uzh_summary['n_real_total']}; "
              f"abstention {100*uzh_summary['abstention']:.1f}%"]
    if uzh_summary["windows_by_sequence"]:
        lines += ["", "| sequence | n | median window s | min | max | samples |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for s, v in uzh_summary["windows_by_sequence"].items():
            lines.append(f"| {s} | {v['n']} | {v['median_s']:.3f} | {v['min_s']:.3f} | "
                         f"{v['max_s']:.3f} | {v['median_samples']:.0f} |")
    if uzh_summary["rates_tii_band"]:
        lines += ["", f"TII band unchanged: {uzh_summary['rates_tii_band']}",
                  f"UZH LOSO: {uzh_summary['rates_uzh_loso']}"]
    if coh:
        lines += ["", f"UZH coherence at 0.10 s: FRR corr {coh['FRR_corr']:.3f}, "
                  f"sign {coh['FRR_sign']:.3f}, n={coh['n_real']}"]
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
