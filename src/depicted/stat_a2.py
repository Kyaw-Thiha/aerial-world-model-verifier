"""The current verdict statistic (Track A2), with the mount as an input instead of a parameter.

Track A2's scorer (`src/score/verdict.py::score_clip`) reconstructs the camera-to-body
rotation from TII's *recorded uptilt* and fits a constant offset to it per clip, because TII
publishes only one number per mount. On UZH-FPV the mount is published as a full rotation
(Kalibr ``T_cam_imu``), so the uptilt-offset nuisance has no reason to exist and the mount is
passed in directly. Everything else is copied from A2 unchanged: the residual is the angle
between the depicted body z axis (gravity + path yaw) and the required thrust axis, the score
is its median over testable frames, and the abstention rule is A2's.

The one fitted nuisance kept is the **time lag**, because it is an evaluation nuisance (frame
timestamps versus the pixel content) that no mount publishes.

A unit test checks that, fed a TII-style uptilt mount and a zero offset grid, this reduces to
``src.score.verdict.score_clip`` numerically.
"""
from __future__ import annotations

import numpy as np

from src.score.verdict import depicted_axis_world, residual_axis, shift_series, tilt_deg_of

# A2 pre-registration (configs/verdict_preregistration.yaml, read-only on track-a2-gate)
ABSTENTION_MIN_SIGNAL_FRAMES = 12
ABSTENTION_MIN_SIGNAL_FRACTION = 0.20
ABSTENTION_MIN_USABLE_FRACTION = 0.40
SIGNAL_TILT_DEG = 10.0
LAG_BOUNDS_MS = (-150.0, 150.0)
LAG_STEP_MS = 25.0
UPTILT_OFFSET_BOUNDS_DEG = (-7.0, 7.0)
UPTILT_OFFSET_STEP_DEG = 1.0
# A2 report 07, "per-fold threshold spread": drag-free, lag-only mode
FROZEN_THRESHOLD_DEG = 14.73


def rotate_mount_about_body_y(R_cb: np.ndarray, delta_rad: float) -> np.ndarray:
    """Mount-uncertainty parameterisation: pitch the camera in the body frame."""
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return Ry @ np.asarray(R_cb, dtype=np.float64)


def score_clip_mount(up_cam: np.ndarray, imp_axis: np.ndarray, yaw: np.ndarray, t: np.ndarray,
                     R_cb: np.ndarray, implied_tilt_deg: np.ndarray, valid: np.ndarray,
                     uptilt_offsets_deg=(0.0,), lag_grid_ms: np.ndarray | None = None) -> dict:
    """A2's clip score with a published mount; returns the score and the abstention decision."""
    up_cam = np.asarray(up_cam, dtype=np.float64)
    imp_axis = np.asarray(imp_axis, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) if valid is not None else np.ones(len(t), bool)
    finite = np.isfinite(up_cam).all(axis=1) & np.isfinite(imp_axis).all(axis=1)
    signal = finite & (np.asarray(implied_tilt_deg) >= SIGNAL_TILT_DEG)

    n_signal = int(signal.sum())
    frac_signal = n_signal / max(len(t), 1)
    frac_valid = float((finite & valid).sum()) / max(len(t), 1)
    reasons = []
    if n_signal < ABSTENTION_MIN_SIGNAL_FRAMES:
        reasons.append(f"signal_frames {n_signal} < {ABSTENTION_MIN_SIGNAL_FRAMES}")
    if frac_signal < ABSTENTION_MIN_SIGNAL_FRACTION:
        reasons.append(f"testable_fraction {frac_signal:.2f} < {ABSTENTION_MIN_SIGNAL_FRACTION}")
    if frac_valid < ABSTENTION_MIN_USABLE_FRACTION:
        reasons.append(f"usable_depicted {frac_valid:.2f} < {ABSTENTION_MIN_USABLE_FRACTION}")
    out = {"n_frames": int(len(t)), "n_signal": n_signal, "frac_signal": frac_signal,
           "frac_usable": frac_valid, "abstain_reasons": reasons}

    lags = (np.arange(LAG_BOUNDS_MS[0], LAG_BOUNDS_MS[1] + 1e-9, LAG_STEP_MS) / 1000.0
            if lag_grid_ms is None else np.asarray(lag_grid_ms, dtype=np.float64) / 1000.0)
    if n_signal == 0:
        out.update({"testable": False, "score": None, "nuisances": None})
        out["abstain_reasons"] = reasons + ["no testable frames"]
        return out

    best = {"loss": np.inf}
    for doff in uptilt_offsets_deg:
        R = rotate_mount_about_body_y(R_cb, np.deg2rad(float(doff)))
        dep = depicted_axis_world(up_cam, R, yaw)
        for lag in lags:
            imp = shift_series(imp_axis, t, float(lag))
            r = residual_axis(dep, imp)
            loss = float(np.nanmedian(r[signal]))
            if loss < best["loss"]:
                best = {"loss": loss, "uptilt_offset_deg": float(doff),
                        "lag_ms": float(lag * 1000.0)}
    R = rotate_mount_about_body_y(R_cb, np.deg2rad(best["uptilt_offset_deg"]))
    dep = depicted_axis_world(up_cam, R, yaw)
    imp = shift_series(imp_axis, t, best["lag_ms"] / 1000.0)
    r_axis = residual_axis(dep, imp)
    use = signal & valid & np.isfinite(r_axis)
    out.update({
        "nuisances": best,
        "testable": (not reasons) and int(use.sum()) > 0,
        "n_scored": int(use.sum()),
        "score": float(np.median(r_axis[use])) if use.any() else None,
        "score_p75": float(np.percentile(r_axis[use], 75)) if use.any() else None,
        "tilt_dep_median": float(np.median(tilt_deg_of(dep)[use])) if use.any() else None,
        "tilt_imp_median": float(np.median(tilt_deg_of(imp)[use])) if use.any() else None,
    })
    if not out["testable"]:
        out["abstain_reasons"] = reasons + (["no usable testable frames"] if not use.any() else [])
    return out
