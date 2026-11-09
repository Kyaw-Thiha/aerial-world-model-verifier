"""Task 0 re-statement of the ANWM excitation audit on ONE pooled population per cell.

Reads the same 33 test trajectories (data/raw/anwm/*.npz); writes
results/anwm_excitation/audit_corrected.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from src.generation.audit import (heading_statistics, per_step_tilt, per_trajectory_median_spread,
                                  pooled_stats, smoothed_tilt)

ROOT = Path(__file__).resolve().parents[1]
K_D = 0.295
DTS = (0.25, 0.667, 1.0)


def main() -> int:
    zs = [np.load(f) for f in sorted(glob.glob(str(ROOT / "data/raw/anwm/*.npz")))]
    pts = [z["point"].astype(float) for z in zs]
    out = {"n_traj": len(pts), "k_d": K_D, "cells": {}}
    for dt in DTS:
        cell = {}
        for kname, k in (("drag_free", 0.0), ("k_d", K_D)):
            ps = [per_step_tilt(p, dt, k) for p in pts]
            sg = [smoothed_tilt(p, dt, k) for p in pts]
            cell[kname] = {"per_step": pooled_stats(ps), "smoothed": pooled_stats(sg),
                           "per_step_median_spread": per_trajectory_median_spread(ps)}
        out["cells"][str(dt)] = cell
    hs = [heading_statistics(p, z["yaw"]) for p, z in zip(pts, zs)]
    def q(a):
        a = np.concatenate(a)
        return {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                "frac_gt_30": float(np.mean(a > 30))}
    out["heading"] = {"yaw_change_deg": q([h["yaw_change"] for h in hs]),
                      "travel_dir_change_deg": q([h["travel_dir_change"] for h in hs])}
    p = ROOT / "results/anwm_excitation/audit_corrected.json"
    p.write_text(json.dumps(out, indent=1))
    for dt, c in out["cells"].items():
        for kn in ("drag_free", "k_d"):
            s, g = c[kn]["per_step"], c[kn]["smoothed"]
            print(f"dt={dt:>6} {kn:9s} per-step n={s['n']} med={s['median']:.2f} p90={s['p90']:.1f} "
                  f">10={s['frac_gt']:.3f} event_frac={s['event_steps']['frac_of_steps']:.3f} "
                  f"event_med={s['event_steps']['median']:.1f} | smoothed med={g['median']:.2f} "
                  f"p90={g['p90']:.1f} >10={g['frac_gt']:.3f}")
    print(json.dumps(out["heading"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
