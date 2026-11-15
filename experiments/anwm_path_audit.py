"""Task 6: path-only required-tilt audit for the three input regimes (no pixels).

For every regime: fraction of steps above 10 deg, median and p90, per-step and SG-smoothed,
at dt = 0.25 / 0.667 / 1.0 s, on one pooled population per cell. Also:
  * the smallest dt at which Regime A's commanded path is consistent with level frames,
  * how much Regime B rescaling changes the required tilt (raw window vs rescaled).
Output: results/anwm_path_audit/audit.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.generation.audit import pooled_stats
from src.generation.paths import (K_D, PathSpec, design, from_mocap, from_native)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/anwm_path_audit"
DTS = (0.25, 2.0 / 3.0, 1.0)
FLIGHTS = ["flight-01a-ellipse", "flight-07a-lemniscate", "flight-13a-trackRATM",
           "flight-01p-ellipse", "flight-07p-lemniscate", "flight-13p-trackRATM",
           "flight-16p-trackRATM"]
MANEUVERS = ("cruise", "turn", "slalom", "accel_brake", "climb_turn")
LEVEL_FRAC = 0.05  # "consistent with level frames": under 5% of steps above 10 deg


def cell(specs: list[PathSpec], dt: float, estimator: str, k_d: float) -> dict:
    return pooled_stats([s.tilt(dt, k_d, estimator) for s in specs])


def regime_cells(specs: list[PathSpec]) -> dict:
    out = {}
    for dt in DTS:
        out[f"{dt:.3f}"] = {f"{est}_{kn}": cell(specs, dt, est, k)
                            for est in ("per_step", "smoothed")
                            for kn, k in (("drag_free", 0.0), ("k_d", K_D))}
    return out


def smallest_consistent_dt(specs: list[PathSpec], estimator: str, k_d: float) -> dict:
    grid = np.geomspace(0.1, 2.0, 40)
    for dt in grid:
        st = cell(specs, dt, estimator, k_d)
        if st["frac_gt"] <= LEVEL_FRAC:
            return {"dt_s": float(dt), "frac_gt": st["frac_gt"], "median": st["median"],
                    "estimator": estimator, "k_d": k_d, "criterion": f"frac>10deg <= {LEVEL_FRAC}"}
    return {"dt_s": None, "estimator": estimator, "k_d": k_d,
            "criterion": f"frac>10deg <= {LEVEL_FRAC}", "note": "not reached on 0.1-2.0 s grid"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"k_d": K_D, "dts": list(DTS), "maneuvers": MANEUVERS}

    # ---- Regime A: released native trajectories
    files = sorted(glob.glob(str(ROOT / "data/raw/anwm/*.npz")))
    A = [from_native(f, 0, 17) for f in files]
    res["A_native"] = {"n_traj": len(A), "cells": regime_cells(A),
                       "smallest_consistent_dt": {
                           f"{est}_k{k:.3f}": smallest_consistent_dt(A, est, k) for est, k in
                           (("per_step", 0.0), ("smoothed", 0.0), ("per_step", K_D),
                            ("smoothed", K_D))}}
    res["A_native"]["per_trajectory_peak_deg"] = [s.peak_tilt(1.0, K_D) for s in A]

    # ---- Regime B: TII mocap, rescaled
    B, changed = [], []
    B_relaxed = []  # explicit off-envelope variant: yaw cap 60 deg/s, speed cap unchanged
    for flight in FLIGHTS:
        hits = list((ROOT / "data/raw/tii").glob(f"*/mocap_{flight}.csv"))
        if not hits:
            print("missing mocap", flight)
            continue
        c = load_tii(hits[0])
        R = Rotation.from_quat(c.quat).as_matrix()
        # heading from the body x-axis projected on the world xy plane: well defined at any
        # pitch, unlike a ZYX Euler yaw (racing attitude sits near the gimbal-lock singularity)
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        dur = float(c.t[-1] - c.t[0])
        for frac in (0.1, 0.35, 0.6, 0.85):
            t0 = float(c.t[0] + frac * dur)
            win_s = min(30.0, dur * (1 - frac))
            try:
                b = from_mocap(c.pos, yaw, c.t, t0, win_s,
                               name=f"B:{flight}:{frac}")
            except Exception as e:
                print("skip", flight, frac, e)
                continue
            if b.meta["raw_peak_h_ms"] < 1.0:
                # hovering / degenerate window: no speed, so no useful required-tilt signal
                print("skip slow", b.name, b.meta["raw_peak_h_ms"])
                continue
            if len(b.pos) < 17:
                print("skip short", b.name, len(b.pos))
                continue
            B.append(b)
            try:
                br = from_mocap(c.pos, yaw, c.t, t0, win_s, name=f"B_relaxed:{flight}:{frac}",
                                yaw_rate_max=60.0)
                if len(br.pos) >= 17 and br.meta["raw_peak_h_ms"] >= 1.0:
                    B_relaxed.append(br)
            except Exception as e:
                print("skip relaxed", flight, frac, e)
            # same window, *not* rescaled, sampled on the same 1 s grid: the comparison that
            # isolates the effect of rescaling at matched dt
            grid = np.arange(0.0, win_s + 1e-9, 1.0)
            raw = PathSpec(name=f"B_raw:{flight}:{frac}", regime="B",
                           pos=np.column_stack([np.interp(grid, c.t, c.pos[:, k]) for k in range(3)]),
                           yaw=np.interp(grid, c.t, yaw), nominal_dt=1.0)
            changed.append({"name": b.name, "lam": b.meta["length_scale_lam"],
                            "tau": b.meta["time_scale_tau"],
                            "raw_peak_h_ms": b.meta["raw_peak_h_ms"],
                            "peak_tilt_1s_raw": raw.peak_tilt(1.0, K_D),
                            "peak_tilt_1s_rescaled": b.peak_tilt(1.0, K_D),
                            "ratio_rescaled_over_raw": (b.peak_tilt(1.0, K_D) /
                                                        max(raw.peak_tilt(1.0, K_D), 1e-9)),
                            "lam_over_tau": b.meta["length_scale_lam"] / b.meta["time_scale_tau"]})
    res["B_rescaled"] = {"n": len(B), "cells": regime_cells(B),
                         "rescale_effect": changed,
                         "yaw_cap_deg_s": 15.0}
    res["B_relaxed"] = {"n": len(B_relaxed), "cells": regime_cells(B_relaxed),
                        "yaw_cap_deg_s": 60.0,
                        "note": "explicit off-distribution variant; ANWM's training yaw step is "
                                "15 deg/s, so this must be reported as off-expert input"}

    # ---- Regime C: factorial designs
    C, metas = [], []
    for m in MANEUVERS:
        levels = (None,) if m == "cruise" else (5.0, 10.0, 20.0)
        for lvl in levels:
            for pol in (("course", "fixed") if m in ("turn", "slalom", "climb_turn")
                        else ("course",)):
                try:
                    s = design(m, lvl, 5.0, 17, 1.0, pol)
                except Exception as e:
                    print("skip", m, lvl, pol, e)
                    continue
                C.append(s)
                metas.append({"name": s.name, "maneuver": m, "target": lvl, "policy": pol,
                              "reachable": s.meta.get("reachable_deg_target"),
                              "achieved_1s_drag_free": s.peak_tilt(1.0, 0.0),
                              "achieved_1s_k_d": s.peak_tilt(1.0, K_D),
                              **s.speeds(), "travel_turn_deg": s.travel_turn_deg()})
    res["C_designed"] = {"n": len(C), "cells": regime_cells(C), "designs": metas}

    (OUT / "audit.json").write_text(json.dumps(res, indent=1))
    for name in ("A_native", "B_rescaled", "B_relaxed", "C_designed"):
        print(f"\n== {name} (n={res[name].get('n', res[name].get('n_traj'))})")
        for dt, cells in res[name]["cells"].items():
            for k, v in cells.items():
                print(f"  dt={dt} {k:22s} n={v['n']:5d} med={v['median']:6.2f} "
                      f"p90={v['p90']:6.1f} >10={v['frac_gt']:.3f}")
    if "smallest_consistent_dt" in res["A_native"]:
        print("\nsmallest dt consistent with level frames (regime A):")
        for k, v in res["A_native"]["smallest_consistent_dt"].items():
            print(" ", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
