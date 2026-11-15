"""Path-only tilt audit helpers (no pixels), shared by Task 0 and Task 6.

Population rule (the Task 0 fix): every statistic of one table cell is computed on ONE
pooled array of per-step tilts (all steps of all trajectories, NaNs dropped). Per-trajectory
medians are never averaged: they are bimodal here (0 deg when fewer than half the steps are
transitions, 20+ deg otherwise), so their mean describes no real step.
"""
from __future__ import annotations

import numpy as np

from src.physics.flatness import G0, implied_tilt_from_positions

THRESH_DEG = 10.0
EVENT_DEG = 1.0  # a step with less than this implied tilt is "no commanded acceleration"


def per_step_tilt(points: np.ndarray, dt: float, k_d: float = 0.0, g: float = G0) -> np.ndarray:
    """Tilt (deg) demanded by the commanded piecewise-constant-velocity steps.

    Identical maths to experiments/anwm_excitation_audit.per_step_tilt (checked by a test);
    index i is the boundary between step i and step i+1. Needs positions (T, 3), T >= 3.
    """
    p = np.asarray(points, dtype=np.float64)
    v = np.diff(p, axis=0) / dt
    dv = np.diff(v, axis=0) / dt
    v_mid = 0.5 * (v[1:] + v[:-1])
    u = dv + np.array([0.0, 0.0, g]) + k_d * v_mid
    out = np.full(len(dv), np.nan)
    ok = np.linalg.norm(dv + np.array([0.0, 0.0, g]), axis=1) > 1.0
    out[ok] = np.degrees(np.arctan2(np.linalg.norm(u[ok][:, :2], axis=1), (dv[:, 2] + g)[ok]))
    return out


def smoothed_tilt(points: np.ndarray, dt: float, k_d: float = 0.0,
                  window: int = 11, polyorder: int = 3) -> np.ndarray:
    tilt, _ = implied_tilt_from_positions(np.asarray(points, float), dt, window, polyorder, k_d)
    return tilt


def pooled_stats(tilts: list[np.ndarray], thresh: float = THRESH_DEG) -> dict:
    """Median / p90 / fraction above threshold on one pooled population."""
    a = np.concatenate([np.asarray(t, float).ravel() for t in tilts]) if tilts else np.array([])
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None, "frac_gt": None}
    out = {"n": int(a.size), "median": float(np.median(a)),
           "p90": float(np.percentile(a, 90)), "frac_gt": float(np.mean(a > thresh))}
    # internal consistency: a median above the threshold forces a majority above it
    assert not (out["median"] > thresh and out["frac_gt"] < 0.5), "inconsistent population"
    ev = a[a > EVENT_DEG]
    out["event_steps"] = {"n": int(ev.size), "frac_of_steps": float(ev.size / a.size),
                          "median": float(np.median(ev)) if ev.size else None,
                          "p90": float(np.percentile(ev, 90)) if ev.size else None,
                          "frac_gt": float(np.mean(ev > thresh)) if ev.size else None}
    return out


def per_trajectory_median_spread(tilts: list[np.ndarray]) -> dict:
    """Distribution of per-trajectory medians (what the old table averaged)."""
    m = np.array([np.nanmedian(t) for t in tilts if np.isfinite(t).any()])
    return {"n_traj": int(m.size), "mean_of_medians": float(m.mean()),
            "median_of_medians": float(np.median(m)), "min": float(m.min()), "max": float(m.max()),
            "n_traj_median_gt_thresh": int((m > THRESH_DEG).sum())}


def wrap180(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def heading_statistics(points: np.ndarray, yaw_rad: np.ndarray, min_step_m: float = 0.5) -> dict[str, np.ndarray]:
    """The two different 'heading' quantities, absolute degrees.

    yaw_change: |change of camera yaw| between consecutive frames. Unconstrained by flatness.
    travel_dir_change: |change of horizontal travel direction| between consecutive
        translating steps (steps shorter than min_step_m are skipped; a direction is undefined
        without translation). Only this quantity implies acceleration.
    """
    p = np.asarray(points, float)
    d = np.diff(p[:, :2], axis=0)
    idx = np.where(np.linalg.norm(d, axis=1) > min_step_m)[0]
    ang = np.degrees(np.arctan2(d[idx, 1], d[idx, 0]))
    return {"yaw_change": np.abs(wrap180(np.diff(np.degrees(yaw_rad)))),
            "travel_dir_change": np.abs(wrap180(np.diff(ang)))}
