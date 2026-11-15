"""Window sensitivity of the dense required lean (R2 and the coherence tier).

Track E found the UZH verdict depends on the required-lean window (0.10 s works;
0.37-1.0 s on coarse positions fails).  This script asks the same question on TII dense
mocap, **without changing either frozen config** and without selecting the window on any
test outcome.

For each window in {0.10, 0.185, 0.37, 0.74} s:

* recompute ``lean_req``/``vel_req`` from the 274.73 Hz truth with that Savitzky-Golay
  window (same polyorder 3), sampled at the frame times;
* refit the frozen gain statistic (same nuisance settings) and report the R2/R1
  conformal band, false-reject, detection and abstention;
* also apply the **frozen 0.37 s bands unchanged** to the new gains, so the band shift is
  visible without refitting;
* rerun the coherence tier with the frozen thresholds 0.6197 / 0.2921 and with the
  window's own leave-one-flight-out threshold.

Selection rules that never look at the test edits are evaluated separately:

* ``truth_gain``: on the fit flights only, choose the window whose truth-attitude gain
  (truth lean on required lean) is closest to 1;
* ``physical_0.10``: the Phase 0 pixel-stage reference, fixed;
* ``sample_rate``: ``max(0.10, 6 * median_dt)`` (a sample-count floor, not a fit).

    PYTHONPATH=. pixi run python experiments/d_window.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.verdict import (FitOptions, coherence, fit_bands, fit_gain, gate_kept_from_conf,
                         load_clip)
from src.verdict.abstain import assess
from src.verdict.decide import Bands, JointBand, ScalarBand, flag
from src.verdict.dense import dense_required_lean, truth_lean

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
CONDITIONS = ["real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]
WINDOWS = [0.10, 0.185, 0.37, 0.74]


def flight_type(flight: str) -> str:
    if flight in TRACK:
        return "track"
    return "piloted" if "p" in flight.split("-")[1] else "autonomous"


def frozen_bands(path: Path) -> Bands:
    cfg = yaml.safe_load(path.read_text())
    b = cfg.get("bands", {})
    out = Bands()
    if "scalar" in b:
        s = b["scalar"]
        out.scalar = ScalarBand(float(s["lo"]), float(s["hi"]), 0.05, int(s.get("n", 0)))
    else:
        # the frozen file carries the selected R2 band only; the D02 R1 band is recorded
        # here so the R1 sensitivity has a reference rather than an undefined band.
        out.scalar = ScalarBand(0.5430, 0.9364, 0.05, 57)
    if "axis" in b:
        a = b["axis"]
        out.axis = JointBand(tuple(a.get("keys", ("gx", "gy"))),
                             {k: float(v) for k, v in a["center"].items()},
                             {k: float(v) for k, v in a["scale"].items()},
                             float(a["threshold"]), 0.05, int(a.get("n", 0)))
    return out


def real_testable(clips, results, abstain_cfg, gate) -> dict:
    out = {}
    for (flight, cid, cond), clip in clips.items():
        if cond != "real":
            continue
        res = results.get((flight, cid, "real"))
        if res is None:
            out[(flight, cid)] = {"testable": False, "reasons": ["no fit"]}
            continue
        gk = gate_kept_from_conf(clip.depicted_conf, gate)
        ab = assess(res, abstain_cfg, gate_kept=gk)
        out[(flight, cid)] = {"testable": ab.testable, "reasons": ab.reasons,
                              "gate_kept": gk}
    return out


def bands_for(results, testable, flights, rule, alpha=0.05) -> Bands:
    fit = [r for (f, cid, cond), r in results.items()
           if cond == "real" and f in flights and testable.get((f, cid), {}).get("testable")]
    return fit_bands(fit, rule, alpha)


def rates(results, testable, rule, alpha=0.05, bands=None) -> dict:
    """Leave-one-flight-out rates.  ``bands`` fixed means apply it unchanged."""
    frr, det = [], {c: [] for c in EDITS}
    for flight in FLIGHTS:
        b = bands if bands is not None else bands_for(
            results, testable, [f for f in FLIGHTS if f != flight], rule, alpha)
        for (f, cid, cond), r in results.items():
            if f != flight or not testable.get((f, cid), {}).get("testable"):
                continue
            if cond == "real":
                frr.append(bool(flag(r, rule, b)))
            elif cond in EDITS:
                det[cond].append(bool(flag(r, rule, b)))
    return {"frr": float(np.mean(frr)) if frr else float("nan"), "n_real": len(frr),
            **{f"det_{c}": float(np.mean(det[c])) if det[c] else float("nan")
               for c in EDITS},
            **{f"n_{c}": len(det[c]) for c in EDITS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/d_window")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    out_dir = ROOT / args.out / "d01"
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_cfg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    coh_cfg = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
    tau_c = float(coh_cfg["thresholds"]["tau_correlation"])
    tau_s = float(coh_cfg["thresholds"]["tau_sign"])
    frozen = frozen_bands(ROOT / "configs/gain_frozen.yaml")

    clips, clip_meta = {}, {}
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        c = load_clip(path)
        clips[(c.flight, c.clip_id, c.condition)] = c
    clip_table = json.loads(
        (ROOT / "data/processed/p3a/p3a_v1/clips.json").read_text())["clips"]
    from experiments.d_eval import load_results, real_testable
    d02 = load_results(ROOT / "results/d_eval/d02/primary_results.json")
    fixed_testable = real_testable(clips, d02, gain_cfg["abstention"], gate)
    for c in clip_table:
        clip_meta[(c["flight"], c["clip_id"])] = c

    # dense arrays once per flight and window (truth lean once)
    dense = {}
    truth = {}
    for flight in FLIGHTS:
        from src.datasets.tii_video import flight_parts
        from scipy.spatial.transform import Rotation
        control, _ = flight_parts(flight)
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz",
                       allow_pickle=False)
        yaw = Rotation.from_matrix(
            Rotation.from_quat(spec["quat"]).as_matrix()).as_euler("ZYX")[:, 0]
        truth[flight] = truth_lean(spec["quat"], yaw)
        for w in WINDOWS:
            dense[(flight, w)] = dense_required_lean(
                ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv", spec["t"],
                window_s=w)

    summary = {"windows": {}, "selection": {}}
    for w in WINDOWS:
        print(f"window {w:.3f} s", flush=True)
        # window-specific clips carry the new required side; depicted is untouched
        wclips, results = {}, {}
        for (flight, cid, cond), clip in clips.items():
            if cond not in CONDITIONS:
                continue
            meta = clip_meta[(flight, cid)]
            lo, hi = meta["index"]
            sl = slice(int(lo), int(hi))
            d = dense[(flight, w)]
            cw = replace(clip, lean_req=d.l_req0[sl].copy(),
                         vel_req=d.vel_req[sl].copy(),
                         tilt_req_deg=d.tilt_deg[sl].copy())
            wclips[(flight, cid, cond)] = cw
            results[(flight, cid, cond)] = fit_gain(
                cw, FitOptions(bootstrap=True, n_boot=args.n_boot))
        # main comparison: the frozen d02 population, so only the window changes
        testable = fixed_testable
        # what this window would abstain on if it were deployed
        testable_w = real_testable(wclips, results, gain_cfg["abstention"], gate)
        n_test_w = sum(1 for v in testable_w.values() if v["testable"])
        bands_refit = {rule: bands_for(results, testable, FLIGHTS, rule)
                       for rule in ("R1_scalar_g1", "R2_axis_gains")}
        entry = {
            "window_s": w,
            "native_samples": int(round(w / 0.0036399)),
            "n_testable_fixed": sum(1 for v in testable.values() if v["testable"]),
            "n_testable_window": n_test_w,
            "abstention_window": 1.0 - n_test_w / max(len(testable_w), 1),
            "testable_by_flight_window": {
                f: {"n": sum(1 for (ff, _), v in testable_w.items() if ff == f),
                    "testable": sum(1 for (ff, _), v in testable_w.items()
                                    if ff == f and v["testable"])}
                for f in FLIGHTS},
            "by_type": {},
            "bands_refit": {},
            "rates_refit": {},
            "rates_frozen_band": {},
            "coherence": {},
        }
        for rule, bands in bands_refit.items():
            entry["bands_refit"][rule] = {
                "scalar": bands.scalar.as_dict() if bands.scalar else None,
                "axis": bands.axis.as_dict() if bands.axis else None,
            }
            entry["rates_refit"][rule] = rates(results, testable, rule, 0.05)
            entry["rates_frozen_band"][rule] = rates(results, testable, rule, 0.05,
                                                     bands=frozen)
            entry.setdefault("rates_window_refit", {})[rule] = rates(
                results, testable_w, rule, 0.05)
        entry["frozen_band"] = {
            "scalar": frozen.scalar.as_dict() if frozen.scalar else None,
            "axis": frozen.axis.as_dict() if frozen.axis else None}
        for typ in ("piloted", "autonomous", "track"):
            fs = [f for f in FLIGHTS if flight_type(f) == typ]
            bands = bands_for(results, testable, fs, "R2_axis_gains") if fs else None
            if bands is None:
                continue
            # evaluate within the type's flights against a band from the other flights
            frr, det = [], {c: [] for c in EDITS}
            for f in fs:
                b = bands_for(results, testable, [x for x in FLIGHTS if x != f],
                              "R2_axis_gains")
                for (ff, cid, cond), r in results.items():
                    if ff != f or not testable.get((ff, cid), {}).get("testable"):
                        continue
                    if cond == "real":
                        frr.append(bool(flag(r, "R2_axis_gains", b)))
                    elif cond in EDITS:
                        det[cond].append(bool(flag(r, "R2_axis_gains", b)))
            entry["by_type"][typ] = {
                "n": len(frr), "frr": float(np.mean(frr)) if frr else float("nan"),
                **{f"det_{c}": float(np.mean(det[c])) if det[c] else float("nan")
                   for c in EDITS}}

        # coherence: same lag as the window's gain fit; frozen thresholds and LOFO tau
        coh_rows = []
        for (flight, cid, cond), cw in wclips.items():
            if cond not in CONDITIONS:
                continue
            if not testable.get((flight, cid), {}).get("testable"):
                continue
            coh = coherence(cw, lag_ms=results[(flight, cid, cond)].lag_ms)
            coh_rows.append({"flight": flight, "clip_id": cid, "condition": cond,
                             "type": flight_type(flight),
                             "r_min": coh.r_min, "s_min": coh.s_min,
                             "correlation_score": coh.correlation_score,
                             "sign_score": coh.sign_score})
        cdf = pd.DataFrame(coh_rows)
        real = cdf[cdf.condition == "real"]
        entry["coherence"]["frozen_thresholds"] = {
            "FRR_corr": float((real.correlation_score > tau_c).mean()),
            "FRR_sign": float((real.sign_score > tau_s).mean()),
            "det_corr": {c: float((cdf[cdf.condition == c].correlation_score > tau_c).mean())
                         for c in EDITS},
            "det_sign": {c: float((cdf[cdf.condition == c].sign_score > tau_s).mean())
                         for c in EDITS},
            "r_min_median": float(real.r_min.median()),
        }
        # LOFO threshold shift
        fold_tau = {}
        frr_c, det_c = [], {c: [] for c in EDITS}
        for f in FLIGHTS:
            fit = real[real.flight != f]
            tau = float(np.quantile(fit.correlation_score, 0.95)) if len(fit) >= 8 else np.nan
            fold_tau[f] = tau
            if not np.isfinite(tau):
                continue
            for _, row in cdf[cdf.flight == f].iterrows():
                if row.condition == "real":
                    frr_c.append(bool(row.correlation_score > tau))
                elif row.condition in EDITS:
                    det_c[row.condition].append(bool(row.correlation_score > tau))
        entry["coherence"]["lofo"] = {
            "tau_per_fold": fold_tau,
            "FRR": float(np.mean(frr_c)) if frr_c else float("nan"),
            "det": {c: float(np.mean(det_c[c])) if det_c[c] else float("nan")
                    for c in EDITS},
        }
        cdf.to_csv(out_dir / f"coherence_w{w:.3f}.csv", index=False)
        summary["windows"][f"{w:.3f}"] = entry

    # ---- selection rules that never look at the test edits
    for fold in FLIGHTS:
        fit_flights = [f for f in FLIGHTS if f != fold]
        scores = {}
        for w in WINDOWS:
            errs = []
            for flight in fit_flights:
                spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz",
                               allow_pickle=False)
                tr, _ = truth[flight]
                d = dense[(flight, w)]
                for meta in [m for m in clip_table if m["flight"] == flight]:
                    key = (flight, meta["clip_id"], "real")
                    if key not in clips:
                        continue
                    lo, hi = meta["index"]
                    sl = slice(int(lo), int(hi))
                    cw = replace(clips[key], lean_req=d.l_req0[sl],
                                 vel_req=d.vel_req[sl], tilt_req_deg=d.tilt_deg[sl])
                    # truth attitude as the "depicted" side: a formula check
                    tw = replace(cw, depicted_lean=tr[sl])
                    r = fit_gain(tw, FitOptions(lag_search=False, bootstrap=False))
                    if np.isfinite(r.gain):
                        errs.append(abs(r.gain - 1.0))
            scores[f"{w:.3f}"] = float(np.mean(errs)) if errs else float("nan")
        best = min(scores, key=lambda k: scores[k])
        summary["selection"][fold] = {"scores": scores, "truth_gain_best": best,
                                      "physical_0.10": "0.100",
                                      "sample_rate": "0.100"}

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    lines = ["# Required-lean window sensitivity (TII, frozen statistic unchanged)", "",
             "| window s | samples | testable (fixed) | window abstain | R2 band (refit) | R2 FRR | removal | mirror | x0.5 | x1.5 | yaw |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, e in summary["windows"].items():
        b = e["bands_refit"]["R2_axis_gains"]["axis"]
        band = f"{b['center']['gx']:.3f}/{b['center']['gy']:.3f} tau {b['threshold']:.3f}" if b else "-"
        r = e["rates_refit"]["R2_axis_gains"]
        lines.append(f"| {k} | {e['native_samples']} | {e['n_testable_fixed']} | "
                     f"{100*e['abstention_window']:.1f}% | {band} | {r['frr']:.3f} | "
                     f"{r['det_tilt_removal']:.3f} | {r['det_mirror_roll']:.3f} | "
                     f"{r['det_scale_0.5']:.3f} | {r['det_scale_1.5']:.3f} | "
                     f"{r['det_yaw_control']:.3f} |")
    lines += ["", "Track flights: 3 testable under 0.37 s; **not validated** at any window.", ""]
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
