"""Task 4: OLS versus total least squares (Deming) for the gain.

The comparison is run on the same 57 testable real clips and their paired edits, with the
same conformal band procedure, for four methods:

* OLS on the v1 required lean (12 Hz differentiation; the frozen D01 statistic),
* Deming on the v1 required lean, with the error-variance ratio measured from the
  coarse-vs-dense difference (x) and the depicted-vs-truth difference (y),
* OLS on the v2 dense `lean_req` (the current statistic),
* Deming on the v2 dense `lean_req` (x error negligible; should approach OLS).

The yaw-control rate is printed beside every detection rate, and track flights are marked
not validated.

    PYTHONPATH=. pixi run python experiments/d_deming.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.score.verdict import wilson_interval
from src.verdict import load_clip
from src.verdict.deming import deming_gain
from src.verdict.fit import _shift_series
from src.verdict.lean import required_lean

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
CONDITIONS = ["real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]


def flight_type(flight: str) -> str:
    if flight in TRACK:
        return "track"
    return "piloted" if "p" in flight.split("-")[1] else "autonomous"


def _scalar_band(values, alpha=0.05):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.quantile(v, [alpha / 2, 1 - alpha / 2]))


def _flag(value, band):
    return bool(np.isfinite(value) and (value < band[0] or value > band[1]))


def _evaluate(df: pd.DataFrame, testable: dict, column: str, alpha=0.05):
    real = df[df.condition == "real"]
    frr, det = [], {c: [] for c in EDITS}
    per_fold = {}
    for flight in FLIGHTS:
        fit = real[(real.flight != flight)]
        band = _scalar_band(fit[column].to_numpy(), alpha)
        per_fold[flight] = band
        sub = df[(df.flight == flight)]
        for _, row in sub.iterrows():
            if not testable.get((row.flight, row.clip_id), {}).get("testable"):
                continue
            if row.condition == "real":
                frr.append(_flag(row[column], band))
            elif row.condition in EDITS:
                det[row.condition].append(_flag(row[column], band))
    pooled = _scalar_band(real[column].to_numpy(), alpha)
    out = {"band_pooled": list(pooled), "band_per_fold": per_fold,
           "frr": float(np.mean(frr)) if frr else float("nan"),
           "frr_wilson": list(wilson_interval(int(np.sum(frr)), len(frr)))
           if frr else [float("nan")] * 2, "n_real": len(frr)}
    for c in EDITS:
        out[f"det_{c}"] = float(np.mean(det[c])) if det[c] else float("nan")
        out[f"det_{c}_wilson"] = list(wilson_interval(int(np.sum(det[c])), len(det[c]))) \
            if det[c] else [float("nan")] * 2
        out[f"n_{c}"] = len(det[c])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/track_d/clips")
    ap.add_argument("--run-v1", default="results/d_eval/d01")
    ap.add_argument("--run-v2", default="results/d_eval/d02")
    ap.add_argument("--decomposition", default="results/d_decomposition/d01/decomposition.csv")
    ap.add_argument("--out", default="results/d_deming")
    args = ap.parse_args()

    from experiments.d_eval import load_results, real_testable
    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    r1 = load_results(ROOT / args.run_v1 / "primary_results.json")
    r2 = load_results(ROOT / args.run_v2 / "primary_results.json")
    clips = {}
    for path in sorted((ROOT / args.clips).glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        clips[(clip.flight, clip.clip_id, clip.condition)] = clip
    testable = real_testable(clips, r2, prereg["abstention"], gate)

    dec = pd.read_csv(ROOT / args.decomposition)
    sigma = {(row.flight, row.clip_id): (row.sigma_x, row.sigma_y)
             for row in dec.itertuples()}

    rows = []
    for (flight, cid, cond), clip in clips.items():
        if cond not in CONDITIONS or not testable.get((flight, cid), {}).get("testable"):
            continue
        if (flight, cid, cond) not in r2:
            continue
        lag = r2[(flight, cid, cond)].lag_ms
        sx, sy = sigma.get((flight, cid), (float("nan"), float("nan")))
        req1 = required_lean(clip.pos, clip.yaw, clip.t,
                             window_s=prereg["required_lean"]["window_s"],
                             polyorder=prereg["required_lean"]["polyorder"])
        dict_row = {"flight": flight, "type": flight_type(flight), "clip_id": cid,
                    "condition": cond, "lag_ms": lag}
        for label, l_req, w, mask, sig_x in (
                ("v1", req1.l_req0, req1.velocity_reg, req1.mask, sx),
                ("v2", clip.lean_req, clip.vel_req, None, 1e-4)):
            lr = _shift_series(l_req, clip.t, lag / 1000.0)
            vv = _shift_series(w, clip.t, lag / 1000.0)
            ok = (np.isfinite(clip.depicted_lean).all(axis=1)
                  & np.isfinite(lr).all(axis=1) & np.isfinite(vv).all(axis=1))
            if mask is not None:
                ok &= mask
            if ok.sum() < 24:
                dict_row[f"{label}_deming"] = float("nan")
                dict_row[f"{label}_ols_deming"] = float("nan")
                continue
            d = deming_gain(clip.depicted_lean[ok], lr[ok], vv[ok],
                            sigma_x=float(sig_x), sigma_y=float(sy))
            dict_row[f"{label}_deming"] = d.gain
            dict_row[f"{label}_ols_deming"] = d.gain_ols
            dict_row[f"{label}_lambda"] = d.lam
        dict_row["v1_ols"] = r1[(flight, cid, cond)].gain
        dict_row["v2_ols"] = r2[(flight, cid, cond)].gain
        rows.append(dict_row)

    df = pd.DataFrame(rows)
    out_dir = ROOT / args.out / "d01"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "deming_table.csv", index=False)

    metrics = {"n_rows": int(len(df)), "methods": {}, "by_type": {}}
    for column in ("v1_ols", "v1_deming", "v2_ols", "v2_deming"):
        metrics["methods"][column] = _evaluate(df, testable, column)
        real = df[df.condition == "real"][["flight", "type", column]].dropna()
        metrics["by_type"][column] = {}
        for typ in ("piloted", "autonomous", "track"):
            sub = real[real.type == typ]
            metrics["by_type"][column][typ] = {
                "n": int(len(sub)),
                "median": float(sub[column].median()) if len(sub) else float("nan"),
                "iqr": [float(sub[column].quantile(0.25)),
                        float(sub[column].quantile(0.75))] if len(sub) else [float("nan")] * 2,
            }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))

    lines = ["# Track D OLS vs Deming", "",
             "Track flights: 4 testable clips; **not validated**.", "",
             "| method | band | FRR | removal | mirror | x0.5 | x1.5 | yaw control |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for column in ("v1_ols", "v1_deming", "v2_ols", "v2_deming"):
        m = metrics["methods"][column]
        lines.append(f"| {column} | {m['band_pooled'][0]:.3f}-{m['band_pooled'][1]:.3f} | "
                     f"{m['frr']:.3f} | {m['det_tilt_removal']:.3f} | "
                     f"{m['det_mirror_roll']:.3f} | {m['det_scale_0.5']:.3f} | "
                     f"{m['det_scale_1.5']:.3f} | {m['det_yaw_control']:.3f} |")
    lines += ["", "Real-clip gain medians (IQR):", ""]
    for column in ("v1_ols", "v1_deming", "v2_ols", "v2_deming"):
        parts = []
        for typ in ("piloted", "autonomous", "track"):
            v = metrics["by_type"][column][typ]
            parts.append(f"{typ} {v['median']:.3f} ({v['iqr'][0]:.3f}-{v['iqr'][1]:.3f}, "
                         f"n={v['n']})")
        lines.append(f"- {column}: " + "; ".join(parts))
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
