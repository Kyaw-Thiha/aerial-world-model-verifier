"""Is one drag coefficient k_d valid for piloted and autonomous TII flights?

Reuses the Phase 0 derivation (SG window 101, order 3, the window all eight folds
chose) and loss (pooled median residual, bounded fit in [0, 3] 1/s). Fits k_d on:
piloted only, autonomous only, all flights, then reports every group's median
residual under every k_d (cross-application), plus drag-free.
Output: results/anwm_excitation/kd_split.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.phase0_floor import (CONFIG, derive, find_mocap, fit_kd_pooled,
                                      pooled_median)
from src.datasets.tii import load_tii

ROOT = Path(__file__).resolve().parents[1]
W, POLY, G = 101, 3, CONFIG["g"]


def main() -> int:
    flights = CONFIG["flights"]
    root = ROOT / CONFIG["dataset_root"]
    derived = {f: derive(load_tii(find_mocap(root, f)), W, POLY, G) for f in flights}
    groups = {
        "piloted": [f for f in flights if f.endswith("p" ) or "p-" in f],
        "autonomous": [f for f in flights if "a-" in f],
        "all": flights,
    }
    groups["piloted"] = [f for f in flights if f.split("-")[1].endswith("p")]
    groups["autonomous"] = [f for f in flights if f.split("-")[1].endswith("a")]
    out = {"window": W, "bounds": CONFIG["kd_bounds"], "groups": groups, "fit": {}, "cross": {}}
    for name, fl in groups.items():
        out["fit"][name] = fit_kd_pooled(derived, fl, G, CONFIG["kd_bounds"])
    for eval_name, fl in groups.items():
        row = {"drag_free": pooled_median(derived, fl, 0.0, G)}
        for fit_name, k in out["fit"].items():
            row[f"kd_{fit_name}={k:.3f}"] = pooled_median(derived, fl, k, G)
        out["cross"][eval_name] = row
    # per-flight single-flight fits (spread of k_d across individual flights)
    out["per_flight_fit"] = {f: fit_kd_pooled(derived, [f], G, CONFIG["kd_bounds"]) for f in flights}
    # speed-bin dependence: the autonomous ellipses are fast, the piloted slow
    out["per_flight_median_at_k"] = {
        f: {"drag_free": pooled_median(derived, [f], 0.0, G),
            "k=0.295": pooled_median(derived, [f], 0.295, G)} for f in flights}
    p = ROOT / "results/anwm_excitation/kd_split.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("fit", "cross", "per_flight_fit")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
