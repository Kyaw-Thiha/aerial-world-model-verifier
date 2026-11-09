"""Track A2, Task 3a: what tilt does each commanded ANWM path require?

Two path interpolators bracket the answer, because ANWM only ever gives waypoints:

* **lower bound** — a minimum-acceleration (natural cubic) spline through the waypoints.
  It is the smoothest flight that still visits every commanded point, so it demands the
  least tilt any flight through those waypoints could need. A path is called infeasible
  only when even this bound asks for tilt the pixels do not show.
* **upper bound** — the per-step estimator: the whole velocity change between consecutive
  waypoints happens inside one step.

Both are evaluated for dt in a sweep, for k_d = 0 (drag-free: the conservative claim) and
k_d = 0.295 (racing-quad drag), using the verifier's own physics (``src.physics``).

Nothing here looks at pixels. The output is joined to depicted tilt by
``experiments/a2_anwm_score.py``.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_anwm_required.py \\
        --clips data/processed/anwm_clips --out results/a2_anwm
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.physics.flatness import G0, implied_thrust_axis, tilt_deg

ROOT = Path(__file__).resolve().parents[1]
K_DS = (0.0, 0.295)
DTS = (0.25, 0.667, 1.0)
DT_SWEEP = np.round(np.arange(0.10, 5.0001, 0.01), 3)
SAMPLES_PER_STEP = 20


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def lower_bound_series(pos: np.ndarray, dt: float, k_d: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration of the natural cubic spline through ``pos`` at ``dt``."""
    n = len(pos)
    t = np.arange(n) * dt
    cs = CubicSpline(t, pos, bc_type="natural", axis=0)
    tt = np.linspace(t[0], t[-1], (n - 1) * SAMPLES_PER_STEP + 1)
    return cs(tt, 1), cs(tt, 2)


def upper_bound_series(pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-step velocity change: v = Δp/dt, a = Δv/dt at interior waypoints."""
    v = np.diff(pos, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    return 0.5 * (v[:-1] + v[1:]), a


def required_tilt(v: np.ndarray, a: np.ndarray, k_d: float) -> np.ndarray:
    axis = implied_thrust_axis(a, v, k_d, G0)
    return tilt_deg(axis)


def summarise(tilt: np.ndarray) -> dict:
    t = np.asarray(tilt, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return {"median_deg": None, "p90_deg": None, "max_deg": None, "frac_ge10": None}
    return {"median_deg": float(np.median(t)), "p90_deg": float(np.percentile(t, 90)),
            "max_deg": float(np.max(t)), "frac_ge10": float(np.mean(t >= 10.0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/anwm_clips")
    ap.add_argument("--out", default="results/a2_anwm")
    ap.add_argument("--threshold-deg", type=float, default=5.76,
                    help="the Gate 3 operating threshold (pooled, all_flights refit)")
    args = ap.parse_args()

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)
    clip_root = ROOT / args.clips
    index = pd.read_csv(clip_root / "clips_index.csv").set_index("clip_id")

    rows: list[dict] = []
    for mf in sorted(clip_root.glob("*/*/manifest.json")):
        man = json.loads(mf.read_text())
        cid = man["id"]
        pos = np.asarray(man["commanded_path"]["positions"], dtype=np.float64)
        step = float(man["commanded_path"]["step_spacing_m"])
        base = {"clip_id": cid, "regime": man["regime"], "seed": man.get("seed"),
                "step_spacing_m": step, "n_waypoints": int(len(pos)),
                "designed_tilt_deg": man.get("designed_required_tilt_deg"),
                "maneuver": (man.get("meta") or {}).get("maneuver"),
                "policy": (man.get("meta") or {}).get("policy")}
        if cid in index.index:
            base["track_b_required_tilt_1s_drag_free_deg"] = float(
                index.loc[cid, "required_tilt_1s_drag_free_deg"])
            base["track_b_required_tilt_1s_kd_deg"] = float(
                index.loc[cid, "required_tilt_1s_kd_deg"])
        # the three nominal dts, both bounds, both k_d
        for dt in DTS:
            v_lo, a_lo = lower_bound_series(pos, dt, 0.0)
            v_hi, a_hi = upper_bound_series(pos, dt)
            for k_d in K_DS:
                rec = dict(base, dt_s=dt, k_d=k_d)
                rec["lower"] = summarise(required_tilt(v_lo, a_lo, k_d))
                rec["upper"] = summarise(required_tilt(v_hi, a_hi, k_d))
                rows.append(rec)
        # dt sweep for the minimum consistent dt (joined to depicted tilt later)
        for k_d in K_DS:
            sweep = []
            for dt in DT_SWEEP:
                v_lo, a_lo = lower_bound_series(pos, dt, 0.0)
                sweep.append(summarise(required_tilt(v_lo, a_lo, k_d))["median_deg"])
            rows.append(dict(base, dt_s=None, k_d=k_d, sweep_dt=DT_SWEEP.tolist(),
                             sweep_lower_median_deg=sweep))

    (out / "required_tilt.json").write_text(json.dumps(
        {"run_id": run, "git": git_hash(), "config": vars(args), "rows": rows},
        indent=1, default=float))

    # a flat CSV for the three nominal dts
    flat = []
    for r in rows:
        if r.get("dt_s") is None:
            continue
        flat.append({**{k: v for k, v in r.items() if k not in ("lower", "upper")},
                     **{f"lower_{k}": v for k, v in r["lower"].items()},
                     **{f"upper_{k}": v for k, v in r["upper"].items()}})
    pd.DataFrame(flat).to_csv(out / "required_tilt.csv", index=False)

    # Cross-check against Track B's columns at dt = 1 s. Their `required_tilt_1s_*` is the
    # MAXIMUM over per-step tilts (established by matching this run's per-step numbers:
    # median |diff| 0.003 deg), not a median, so the max is the statistic to compare.
    cross = {}
    for k_d, col in ((0.0, "track_b_required_tilt_1s_drag_free_deg"),
                     (0.295, "track_b_required_tilt_1s_kd_deg")):
        chk = [r for r in rows if r.get("dt_s") == 1.0 and r["k_d"] == k_d
               and r.get(col) is not None]
        if not chk:
            continue
        mine_max = np.array([r["upper"]["max_deg"] for r in chk], dtype=np.float64)
        mine_med = np.array([r["upper"]["median_deg"] for r in chk], dtype=np.float64)
        theirs = np.array([r[col] for r in chk])
        dmax = np.abs(mine_max - theirs)
        dmed = np.abs(mine_med - theirs)
        cross[f"k_d_{k_d:g}"] = {
            "n": len(chk), "statistic": "max over steps",
            "median_abs_diff_deg": float(np.median(dmax)),
            "p90_abs_diff_deg": float(np.percentile(dmax, 90)),
            "max_abs_diff_deg": float(np.max(dmax)),
            "median_abs_diff_if_median_were_used_deg": float(np.median(dmed)),
            "n_clips_where_median_differs_by_gt_10deg": int(np.sum(dmed > 10.0)),
        }
    (out / "crosscheck_track_b.json").write_text(json.dumps(cross, indent=2))
    print("cross-check vs Track B (max per step):", json.dumps(cross), flush=True)

    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
