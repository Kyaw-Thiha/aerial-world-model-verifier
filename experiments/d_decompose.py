"""Task 1: decompose the real-flight gain with dense mocap.

Three regressions, per clip, on the same 57 testable real clips:

(a) truth-attitude lean on required lean, with the velocity regressor and a free
    intercept.  This validates the flatness formula and the drag nuisance: the
    acceleration gain should be about 1 and the velocity coefficient about 0.295 1/s.
(b) estimated (GeoCalib) lean on truth-attitude lean, free intercept, no drag.  This is
    the estimator's shrinkage.
(c) the verdict's own gain: estimated lean on required lean.  Both the frozen v1
    statistic (required lean differentiated on the 12 Hz frame grid) and the v2 statistic
    (dense `lean_req`) are reported, so the dense fix can be read against the same chain.

Intervals: per-clip block-bootstrap intervals from `fit_gain`; fleet medians with a
flight-cluster bootstrap (resample flights, not clips).  Per flight type.

    PYTHONPATH=. pixi run python experiments/d_decompose.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.datasets.tii_video import flight_parts
from src.verdict import (FitOptions, dense_required_lean, fit_gain, load_clip,
                         truth_lean)
from src.verdict.lean import required_lean, world_to_yaw_frame
from src.verdict.schema import Clip

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}


def flight_type(flight: str) -> str:
    if flight in TRACK:
        return "track"
    return "piloted" if "p" in flight.split("-")[1] else "autonomous"


def cluster_bootstrap_median(df: pd.DataFrame, column: str, n_boot: int = 4000,
                             seed: int = 20261114) -> dict:
    """Median and 95 % interval resampling whole flights (clips cluster within flight)."""
    flights = sorted(df["flight"].unique())
    values = {f: df.loc[df.flight == f, column].dropna().to_numpy() for f in flights}
    values = {f: v for f, v in values.items() if len(v)}
    if not values:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_clips": 0, "n_flights": 0}
    rng = np.random.default_rng(seed)
    keys = list(values)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample = np.concatenate([values[keys[i]] for i in pick])
        boots.append(np.median(sample))
    return {"median": float(np.median(df[column].dropna())),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "n_clips": int(df[column].notna().sum()), "n_flights": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/track_d/clips")
    ap.add_argument("--run-v1", default="results/d_eval/d01")
    ap.add_argument("--out", default="results/d_decomposition")
    ap.add_argument("--tag", default="d01")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    from experiments.d_eval import load_results, real_testable

    v1 = load_results(ROOT / args.run_v1 / "primary_results.json")
    clips = {}
    for path in sorted((ROOT / args.clips).glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        clips[(clip.flight, clip.clip_id, clip.condition)] = clip
    testable = real_testable(clips, v1, prereg["abstention"], gate)

    out_dir = ROOT / args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for flight in FLIGHTS:
        control, _ = flight_parts(flight)
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz",
                       allow_pickle=False)
        from scipy.spatial.transform import Rotation
        yaw = Rotation.from_matrix(
            Rotation.from_quat(spec["quat"]).as_matrix()).as_euler("ZYX")[:, 0]
        dense = dense_required_lean(
            ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv", spec["t"])
        truth, truth_tilt = truth_lean(spec["quat"], yaw)
        clip_table = json.loads(
            (ROOT / "data/processed/p3a/p3a_v1/clips.json").read_text())["clips"]
        for clip_meta in [c for c in clip_table if c["flight"] == flight]:
            key = (flight, clip_meta["clip_id"], "real")
            if key not in clips or not testable.get((flight, clip_meta["clip_id"]),
                                                    {}).get("testable"):
                continue
            lo, hi = clip_meta["index"]
            sl = slice(int(lo), int(hi))
            clip = clips[key]
            if clip.lean_req is None:
                clip = replace(clip, lean_req=dense.l_req0[sl],
                               vel_req=dense.vel_req[sl],
                               tilt_req_deg=dense.tilt_deg[sl])
            est = clip.depicted_lean
            # (a) truth attitude on the required path
            a_clip = replace(clip, depicted_lean=truth[sl])
            a = fit_gain(a_clip, FitOptions(lag_search=False, bootstrap=True,
                                            n_boot=args.n_boot))
            a_fixed = fit_gain(a_clip, FitOptions(lag_search=False, bootstrap=True,
                                                  n_boot=args.n_boot, drag="fixed"))
            a_free = fit_gain(a_clip, FitOptions(lag_search=False, bootstrap=True,
                                                 n_boot=args.n_boot, drag="free"))
            # (b) estimated attitude on truth attitude (no path, no drag)
            b_clip = replace(clip, lean_req=truth[sl],
                             vel_req=np.zeros_like(truth[sl]),
                             tilt_req_deg=truth_tilt[sl])
            b = fit_gain(b_clip, FitOptions(drag="free", lag_search=True,
                                            bootstrap=True, n_boot=args.n_boot))
            # (c) v2: estimated on dense required path
            c2 = fit_gain(clip, FitOptions(bootstrap=True, n_boot=args.n_boot))
            # (c) v1: frozen d01 value
            c1 = v1[key]

            # noise inputs for the Deming comparison
            req_v1 = required_lean(clip.pos, clip.yaw, clip.t,
                                   window_s=prereg["required_lean"]["window_s"],
                                   polyorder=prereg["required_lean"]["polyorder"])
            ok = (np.isfinite(clip.lean_req).all(axis=1)
                  & np.isfinite(req_v1.l_req0).all(axis=1)
                  & np.isfinite(est).all(axis=1) & np.isfinite(truth[sl]).all(axis=1))
            dx = req_v1.l_req0[ok] - clip.lean_req[ok]
            dy = est[ok] - truth[sl][ok]
            rows.append({
                "flight": flight, "type": flight_type(flight),
                "clip_id": clip_meta["clip_id"],
                "a_gain": a.gain, "a_gain_lo": a.gain_ci[0], "a_gain_hi": a.gain_ci[1],
                "a_drag": a.drag, "a_drag_lo": a.drag_ci[0], "a_drag_hi": a.drag_ci[1],
                "a_fixed_gain": a_fixed.gain, "a_free_gain": a_free.gain,
                "a_r2": a.r2, "a_lag_ms": a.lag_ms,
                "b_gain": b.gain, "b_gain_lo": b.gain_ci[0], "b_gain_hi": b.gain_ci[1],
                "b_lag_ms": b.lag_ms, "b_r2": b.r2,
                "c_v1_gain": c1.gain, "c_v1_lo": c1.gain_ci[0], "c_v1_hi": c1.gain_ci[1],
                "c_v2_gain": c2.gain, "c_v2_lo": c2.gain_ci[0],
                "c_v2_hi": c2.gain_ci[1], "c_v2_drag": c2.drag, "c_v2_r2": c2.r2,
                "c_v2_lag_ms": c2.lag_ms,
                "sigma_x": float(np.sqrt(np.mean(dx ** 2))),
                "sigma_y": float(np.sqrt(np.mean(dy ** 2))),
                "n_used": int(ok.sum()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "decomposition.csv", index=False)
    summary = {"n_clips": int(len(df)), "by_type": {}, "all": {}}
    for column in ("a_gain", "a_drag", "a_fixed_gain", "a_free_gain", "b_gain",
                   "c_v1_gain", "c_v2_gain"):
        summary["all"][column] = cluster_bootstrap_median(df, column)
        summary["all"][column]["iqr"] = [float(df[column].quantile(0.25)),
                                         float(df[column].quantile(0.75))]
        for typ in ("piloted", "autonomous", "track"):
            sub = df[df.type == typ]
            summary["by_type"].setdefault(typ, {})[column] = \
                cluster_bootstrap_median(sub, column)
    summary["correlation"] = {
        "c_v1_vs_a_times_b": float(df[["c_v1_gain"]].assign(
            ab=df.a_gain * df.b_gain).corr().iloc[0, 1]),
        "c_v2_vs_a_times_b": float(df[["c_v2_gain"]].assign(
            ab=df.a_gain * df.b_gain).corr().iloc[0, 1]),
    }
    (out_dir / "decomposition.json").write_text(json.dumps(summary, indent=1,
                                                           default=float))
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
