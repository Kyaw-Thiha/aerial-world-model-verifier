"""Inspect released ANWM trajectory metadata: step size, yaw step, attitude, and dt.

Answers three questions with no model run:

1. What is the metric step between consecutive recorded frames, and how often is
   it a translation versus a pure yaw turn?
2. What roll and pitch do the recorded trajectories carry (i.e. what attitude do
   the training frames depict), and are those columns used in the camera pose?
   (`Data_preprocessing_airvln_16.ipynb` feeds `euler_to_quaternion(yaw, pitch, roll)`
   into `create_pose_matrix`, so yes.)
3. Seconds per step: the release never states one. This script measures the dt
   that makes the *recorded attitude* agree with the attitude *implied* by the
   recorded motion, and reports the implied-tilt distribution for the three
   candidate values (1.0 s from the paper's per-action distances, 0.667 s from
   3.30 m at 5 m/s, 0.25 s from the code default `--input_fps 4`).

Input: `data/raw/anwm/*.npz` written by `scripts/fetch_anwm.py`.
Output: `results/anwm_excitation/inspection.json` plus a printed table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.physics.flatness import G0, implied_tilt_from_positions, tilt_from_roll_pitch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data/raw/anwm"
OUT_DIR = ROOT / "results/anwm_excitation"

CANDIDATE_DT = {"paper_1s": 1.0, "spacing_over_5ms": 3.30 / 5.0, "code_input_fps4": 0.25}
DT_GRID = np.geomspace(0.05, 2.0, 60)
SG_WINDOW = 11
SG_POLY = 3


def _wrap_180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def _implied(pts: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tilt, mask) for one candidate dt, using the shared physics helper."""
    return implied_tilt_from_positions(pts, float(dt), SG_WINDOW, SG_POLY, 0.0, G0)


def inspect_one(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    keys = sorted(z.files)
    pts = np.asarray(z["point"], dtype=np.float64) if "point" in keys else None
    if pts is None:  # fall back to 2-D positions with z from a zero column
        pts = np.column_stack([z["position"], np.zeros(len(z["position"]))])
    yaw = np.asarray(z["yaw"], dtype=np.float64).ravel()
    roll = np.asarray(z["roll"], dtype=np.float64).ravel() if "roll" in keys else None
    pitch = np.asarray(z["pitch"], dtype=np.float64).ravel() if "pitch" in keys else None
    n = len(pts)

    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    step_xy = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    yaw_step = np.degrees(_wrap_180(np.diff(np.degrees(yaw))))

    out = {
        "file": path.name,
        "n_frames": int(n),
        "keys": keys,
        "dt_step_m": {"mean": float(step.mean()), "median": float(np.median(step)),
                      "p10": float(np.percentile(step, 10)), "p90": float(np.percentile(step, 90)),
                      "max": float(step.max())},
        "dt_step_xy_m": {"mean": float(step_xy.mean()), "median": float(np.median(step_xy))},
        "frac_steps_translation": float(np.mean(step > 0.5)),
        "frac_steps_pure_yaw": float(np.mean(step <= 1e-6)),
        "yaw_step_deg": {"median": float(np.median(yaw_step)), "p90": float(np.percentile(yaw_step, 90)),
                         "frac_15deg": float(np.mean(np.abs(np.abs(yaw_step) - 15) < 1.0)),
                         "frac_zero": float(np.mean(np.abs(yaw_step) < 1e-6))},
    }

    if roll is not None and pitch is not None:
        # decide units from magnitude: the notebook copies them straight from the
        # AerialVLN reference path, which is radians
        degrees = bool(np.nanmax(np.abs(np.concatenate([roll, pitch]))) > 4.0) and \
            bool(np.nanmax(np.abs(np.concatenate([roll, pitch]))) > np.pi + 0.1)
        out["attitude_units"] = "deg" if degrees else "rad"
        dep = tilt_from_roll_pitch(roll, pitch, degrees=degrees)
        out["depicted_tilt_deg"] = {
            "median": float(np.median(dep)), "p90": float(np.percentile(dep, 90)),
            "max": float(dep.max()),
            "frac_gt_1deg": float(np.mean(dep > 1.0)),
            "frac_gt_5deg": float(np.mean(dep > 5.0)),
            "frac_gt_10deg": float(np.mean(dep > 10.0)),
        }
        out["roll_deg"] = {"median": float(np.degrees(np.median(roll)) if not degrees else np.median(roll)),
                           "abs_p90": float(np.degrees(np.percentile(np.abs(roll), 90)) if not degrees
                                            else np.percentile(np.abs(roll), 90))}
        out["pitch_deg"] = {"median": float(np.degrees(np.median(pitch)) if not degrees else np.median(pitch)),
                            "abs_p90": float(np.degrees(np.percentile(np.abs(pitch), 90)) if not degrees
                                             else np.percentile(np.abs(pitch), 90))}
    else:
        dep = None

    # implied tilt at the three candidate step durations
    out["candidates"] = {}
    for tag, dt in CANDIDATE_DT.items():
        try:
            tilt, mask = _implied(pts, dt)
            row = {
                "dt_s": dt,
                "implied_tilt_median_deg": float(np.nanmedian(tilt)),
                "implied_tilt_p90_deg": float(np.nanpercentile(tilt, 90)),
                "frac_gt_10deg": float(np.nanmean(tilt > 10.0)),
            }
            if dep is not None:
                m = mask & (tilt > 1.0)
                row["median_abs_diff_vs_depicted_deg"] = (
                    float(np.nanmedian(np.abs(tilt[m] - dep[m]))) if m.sum() > 10 else None)
                row["n_compared"] = int(m.sum())
            out["candidates"][tag] = row
        except ValueError as e:
            out["candidates"][tag] = {"dt_s": dt, "error": str(e)}

    # dt that best explains the recorded attitude
    if dep is not None:
        scores = []
        for dt in DT_GRID:
            try:
                tilt, _ = _implied(pts, dt)
            except ValueError:
                continue
            m = ~np.isnan(tilt) & (np.abs(tilt - dep) < 180)
            if m.sum() < 20:
                continue
            scores.append((float(dt), float(np.nanmedian(np.abs(tilt[m] - dep[m]))), int(m.sum())))
        if scores:
            best = min(scores, key=lambda r: r[1])
            out["best_dt_by_attitude"] = {"dt_s": best[0], "median_abs_diff_deg": best[1],
                                          "n": best[2], "grid": [s[0] for s in scores[::6][:10]]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    files = sorted(IN_DIR.glob("*.npz"))[: args.limit]
    if not files:
        raise SystemExit(f"no npz files in {IN_DIR}; run scripts/fetch_anwm.py --fetch-test first")
    rows = [inspect_one(p) for p in files]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def agg(sub_key):
        vals = [r["dt_step_m"][sub_key] for r in rows
                if "dt_step_m" in r and sub_key in r["dt_step_m"]]
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))} if vals else None

    summary = {
        "n_trajectories": len(rows),
        "step_m_mean": agg("mean"),
        "yaw_step_deg_median_mean": float(np.mean([r["yaw_step_deg"]["median"] for r in rows])),
        "per_trajectory": rows,
    }
    for tag in CANDIDATE_DT:
        vals = [r["candidates"][tag].get("implied_tilt_median_deg") for r in rows
                if "implied_tilt_median_deg" in r["candidates"][tag]]
        summary[f"implied_tilt_median_deg_{tag}"] = float(np.mean(vals)) if vals else None
    best = [r["best_dt_by_attitude"]["dt_s"] for r in rows if "best_dt_by_attitude" in r]
    summary["best_dt_by_attitude"] = {
        "median": float(np.median(best)) if best else None,
        "n_trajectories": len(best),
    }
    (OUT_DIR / "inspection.json").write_text(json.dumps(summary, indent=1))

    print(f"trajectories: {len(rows)}")
    print(f"{'file':44s} {'T':>5} {'step m':>7} {'yaw step':>9} {'dep tilt':>9} "
          f"{'imp@1.0':>8} {'imp@0.67':>9} {'imp@0.25':>9} {'best dt':>8}")
    for r in rows:
        dep = r.get("depicted_tilt_deg", {}).get("median")
        c = r["candidates"]
        b = r.get("best_dt_by_attitude", {}).get("dt_s")
        print(f"{r['file'][:44]:44s} {r['n_frames']:5d} {r['dt_step_m']['mean']:7.2f} "
              f"{r['yaw_step_deg']['median']:9.2f} "
              f"{(f'{dep:9.2f}' if dep is not None else '        -')} "
              f"{c['paper_1s'].get('implied_tilt_median_deg', float('nan')):8.2f} "
              f"{c['spacing_over_5ms'].get('implied_tilt_median_deg', float('nan')):9.2f} "
              f"{c['code_input_fps4'].get('implied_tilt_median_deg', float('nan')):9.2f} "
              f"{(f'{b:8.3f}' if b else '       -')}")
    print("\nbest dt by attitude agreement:", summary["best_dt_by_attitude"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
