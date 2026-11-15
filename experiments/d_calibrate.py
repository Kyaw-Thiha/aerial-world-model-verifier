"""Task 1c: does a monotone calibration curve fitted on fit flights improve detection?

The diagnosis (`experiments/d_shrinkage.py`) finds a roughly **linear** compression on the
roll/lateral axis of piloted flights (slope 0.79, R2 0.99) and an unbiased pitch axis
(slope 1.00).  A monotone calibration is therefore a candidate.  Two variants are tested,
each fitted on the fit flights only and applied leave-one-flight-out:

* **single curve**: one isotonic (PAVA) curve on the pooled |lean| magnitudes of both axes;
* **per-axis curves**: one curve per axis (the honest form of the diagnosis).

For each outer fold the curve is fitted on the real frames of the fit flights, applied to
every clip's depicted lean, and the gain statistic (R1 scalar and R2 axis, frozen
nuisance settings) is refitted.  Bands are conformal on the fit flights and evaluated on
the held-out flight, so false-reject and detection are compared at matched rates.

    PYTHONPATH=. pixi run python experiments/d_calibrate.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.verdict import FitOptions, fit_bands, fit_gain, flag, load_clip

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]
CONDITIONS = ["real"] + EDITS
ALPHAS = np.round(np.arange(0.0, 0.3001, 0.025), 4)


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators, non-decreasing, weighted."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    blocks: list[list[float]] = []   # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out = []
    for v, _, n in blocks:
        out.extend([v] * int(n))
    return np.array(out)


def fit_curve(frames: pd.DataFrame, per_axis: bool, n_bins: int = 12,
              min_count: int = 50) -> dict:
    """Fit monotone magnitude curves; returns {dep_knots, truth_knots} per axis."""
    curves = {}
    axes = (0, 1) if per_axis else ("pooled",)
    for axis in axes:
        if axis == "pooled":
            truth = np.concatenate([np.abs(frames.truth_x), np.abs(frames.truth_y)])
            dep = np.concatenate([np.abs(frames.dep_x), np.abs(frames.dep_y)])
        else:
            name = "x" if axis == 0 else "y"
            truth = np.abs(frames[f"truth_{name}"].to_numpy())
            dep = np.abs(frames[f"dep_{name}"].to_numpy())
        ok = np.isfinite(truth) & np.isfinite(dep)
        truth, dep = truth[ok], dep[ok]
        edges = np.quantile(truth, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        xs, ys, ws = [0.0], [0.0], [max(len(truth) // 20, 1)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (truth >= lo) & (truth < hi) if hi > lo else (truth == lo)
            if m.sum() < min_count:
                continue
            xs.append(float(np.mean(truth[m])))
            ys.append(float(np.mean(dep[m])))
            ws.append(int(m.sum()))
        ys = pava(np.array(ys), np.array(ws))
        xs = np.array(xs)
        # keep strictly increasing inversion knots
        keep = np.concatenate([[True], np.diff(ys) > 1e-12])
        curves[axis] = {"dep_knots": ys[keep].tolist(),
                        "truth_knots": xs[keep].tolist()}
    return curves


def apply_curve(clip, curves: dict):
    dep = np.asarray(clip.depicted_lean, dtype=np.float64).copy()
    out = np.full_like(dep, np.nan)
    if "pooled" in curves:
        mag = np.abs(dep)
        ok = np.isfinite(mag).all(axis=1)
        for i in np.flatnonzero(ok):
            h = np.interp(mag[i], curves["pooled"]["dep_knots"],
                          curves["pooled"]["truth_knots"])
            out[i] = np.sign(dep[i]) * h
    else:
        for axis in (0, 1):
            c = curves[axis]
            col = dep[:, axis]
            ok = np.isfinite(col)
            out[ok, axis] = np.sign(col[ok]) * np.interp(
                np.abs(col[ok]), c["dep_knots"], c["truth_knots"])
    return replace(clip, depicted_lean=out)


def evaluate(rows: list[dict], testable: dict, rule: str, alpha: float) -> dict:
    df = pd.DataFrame(rows)
    real = df[df.condition == "real"]
    frr, det = [], {c: [] for c in EDITS}
    for flight in FLIGHTS:
        fit = [r for r in rows if r["condition"] == "real" and r["flight"] != flight
               and testable.get((r["flight"], r["clip_id"]), {}).get("testable")]
        bands = fit_bands([r["result"] for r in fit], rule, alpha)
        for r in rows:
            if r["flight"] != flight or not testable.get((r["flight"], r["clip_id"]),
                                                         {}).get("testable"):
                continue
            if r["condition"] == "real":
                frr.append(bool(flag(r["result"], rule, bands)))
            elif r["condition"] in EDITS:
                det[r["condition"]].append(bool(flag(r["result"], rule, bands)))
    return {"frr": float(np.mean(frr)) if frr else float("nan"),
            "n_real": len(frr),
            **{f"det_{c}": float(np.mean(det[c])) if det[c] else float("nan")
               for c in EDITS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/d_eval/d02")
    ap.add_argument("--out", default="results/d_calibrate")
    args = ap.parse_args()
    from experiments.d_eval import load_results, real_testable
    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    results = load_results(ROOT / args.run / "primary_results.json")
    clips = {}
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        c = load_clip(path)
        clips[(c.flight, c.clip_id, c.condition)] = c
    testable = real_testable(clips, results, prereg["abstention"], gate)

    frames = pd.read_csv(ROOT / "results/d_shrinkage/d01/frames.csv")
    out_dir = ROOT / args.out / "d01"
    out_dir.mkdir(parents=True, exist_ok=True)

    # uncalibrated reference rows
    reference = [{"flight": f, "clip_id": cid, "condition": cond,
                  "result": res} for (f, cid, cond), res in results.items()
                 if cond in CONDITIONS]
    untested_ref = {rule: evaluate(reference, testable, rule, 0.05)
                    for rule in ("R1_scalar_g1", "R2_axis_gains")}

    summary = {"reference": untested_ref, "calibrated": {}, "matched": {}}
    for mode in ("single", "per_axis"):
        rows_all, fold_curves = [], {}
        for flight in FLIGHTS:
            fit_frames = frames[frames.flight != flight]
            curves = fit_curve(fit_frames, per_axis=(mode == "per_axis"))
            fold_curves[flight] = curves
            fit_key = flight
            for (f, cid, cond), clip in clips.items():
                if cond not in CONDITIONS:
                    continue
                if not testable.get((f, cid), {}).get("testable"):
                    continue
                cal = apply_curve(clip, curves)
                r = fit_gain(cal, FitOptions(bootstrap=False))
                rows_all.append({"flight": f, "clip_id": cid, "condition": cond,
                                 "result": r})
        summary["calibrated"][mode] = {}
        for rule in ("R1_scalar_g1", "R2_axis_gains"):
            summary["calibrated"][mode][rule] = evaluate(rows_all, testable, rule, 0.05)
            # operating curve for the matched-false-reject comparison
            curve = []
            for alpha in ALPHAS:
                curve.append({"alpha": float(alpha),
                              **evaluate(rows_all, testable, rule, float(alpha))})
            summary["calibrated"][mode][rule]["curve"] = curve
        summary.setdefault("fold_curves", {})[mode] = {
            f: c for f, c in fold_curves.items()}
    (out_dir / "calibrate.json").write_text(json.dumps(summary, indent=1, default=float))

    lines = ["# Monotone calibration test", "",
             "Track flights: 3 testable under v2; **not validated**.", "",
             "| method / rule | FRR | removal | mirror | x0.5 | x1.5 | yaw |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for rule in ("R1_scalar_g1", "R2_axis_gains"):
        r = untested_ref[rule]
        lines.append(f"| uncalibrated {rule} | {r['frr']:.3f} | {r['det_tilt_removal']:.3f} | "
                     f"{r['det_mirror_roll']:.3f} | {r['det_scale_0.5']:.3f} | "
                     f"{r['det_scale_1.5']:.3f} | {r['det_yaw_control']:.3f} |")
    for mode in ("single", "per_axis"):
        for rule in ("R1_scalar_g1", "R2_axis_gains"):
            r = summary["calibrated"][mode][rule]
            lines.append(f"| {mode} curve {rule} | {r['frr']:.3f} | "
                         f"{r['det_tilt_removal']:.3f} | {r['det_mirror_roll']:.3f} | "
                         f"{r['det_scale_0.5']:.3f} | {r['det_scale_1.5']:.3f} | "
                         f"{r['det_yaw_control']:.3f} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
