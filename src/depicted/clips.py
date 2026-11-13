"""Clip construction for the UZH verdict test (shared schema, command-path mode).

A clip is a non-overlapping window of prepared frames. The path side is exact (Leica truth),
so the only quantity the pixels supply is the depicted attitude. Everything here is the TII
rule set, unchanged:

* clip length 4 s at the prepared rate (15 Hz -> about 60 frames; TII used 4 s at 12 Hz);
* a clip is testable if at least 12 frames carry a required tilt of at least 10 deg, at
  least 20% of its frames do, and at least 40% of its frames have a usable depicted
  estimate (`src.depicted.stat_a2`);
* the required thrust axis is `a + g e3 + k_d v` normalised, from the truth path
  differentiated on its own 500 Hz grid and sampled at the frame times.
"""
from __future__ import annotations

import numpy as np

from src.depicted.uzh import UzhTruth, dense_kinematics, required_tilt_deg

G0 = 9.81
CLIP_SECONDS = 4.0
SIGNAL_TILT_DEG = 10.0
MIN_SIGNAL_FRAMES = 12
MIN_SIGNAL_FRACTION = 0.20


def clip_bounds(n: int, seconds: float = CLIP_SECONDS, fps: float = 15.0) -> list[tuple[int, int]]:
    """Non-overlapping ``[lo, hi)`` windows of a fixed duration in frames."""
    k = max(1, int(round(seconds * fps)))
    return [(lo, lo + k) for lo in range(0, n - k + 1, k)]


def path_side_with_gt(truth: UzhTruth, gt: np.ndarray, k_d: float = 0.0,
                      window_s: float = 0.10) -> dict:
    """Same, with the dense ground-truth table passed in (N x 8, TUM)."""
    kin = dense_kinematics(gt, window_s=window_s)
    t_gt = kin["t"] - (truth.t_unix[0] + truth.offset_s)
    a = np.column_stack([np.interp(truth.t, t_gt, kin["a"][:, k]) for k in range(3)])
    v = np.column_stack([np.interp(truth.t, t_gt, kin["v"][:, k]) for k in range(3)])
    tilt, axis, ok = required_tilt_deg(a, v, k_d=k_d, g=G0)
    return {"accel": a, "vel": v, "required_tilt_deg": tilt, "required_axis": axis,
            "ok": ok, "speed": np.linalg.norm(v, axis=1), "k_d": k_d,
            "kin_window_s": kin["window_s"]}


def signal_mask(required_tilt_deg: np.ndarray, valid_depicted: np.ndarray | None = None) -> dict:
    """The pre-registered testability counts for one window."""
    sig = np.asarray(required_tilt_deg) >= SIGNAL_TILT_DEG
    n = len(sig)
    counts = {"n_frames": int(n), "n_signal": int(sig.sum()),
              "frac_signal": float(sig.sum() / max(n, 1)),
              "frac_usable": float(np.mean(valid_depicted)) if valid_depicted is not None else 1.0}
    reasons = []
    if counts["n_signal"] < MIN_SIGNAL_FRAMES:
        reasons.append(f"signal_frames {counts['n_signal']} < {MIN_SIGNAL_FRAMES}")
    if counts["frac_signal"] < MIN_SIGNAL_FRACTION:
        reasons.append(f"testable_fraction {counts['frac_signal']:.2f} < {MIN_SIGNAL_FRACTION}")
    if counts["frac_usable"] < 0.40:
        reasons.append(f"usable_depicted {counts['frac_usable']:.2f} < 0.40")
    counts["testable"] = not reasons
    counts["abstain_reasons"] = reasons
    return counts


def lean_from_up_body(up_b: np.ndarray) -> np.ndarray:
    """Schema lean vector ``(cos(phi) sin(theta), -sin(phi))`` from world up in the body.

    ``up_b = (-sin(theta), sin(phi) cos(theta), cos(phi) cos(theta))`` for a ZYX attitude, so
    ``theta = -asin(up_x)`` and ``phi = atan2(up_y, up_z)``. Rows whose up vector is missing
    (estimator failure) stay NaN.
    """
    up_b = np.asarray(up_b, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(up_b, axis=1)
    ok = np.isfinite(n) & (n > 1e-9)
    up = np.full_like(up_b, np.nan)
    up[ok] = up_b[ok] / n[ok, None]
    theta = -np.arcsin(np.clip(up[:, 0], -1.0, 1.0))
    phi = np.arctan2(up[:, 1], up[:, 2])
    return np.column_stack([np.cos(phi) * np.sin(theta), -np.sin(phi)])


def roll_pitch_from_up_body(up_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depicted body roll and pitch (radians) from world up in the body frame (ZYX)."""
    up_b = np.asarray(up_b, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(up_b, axis=1)
    ok = np.isfinite(n) & (n > 1e-9)
    u = np.full_like(up_b, np.nan)
    u[ok] = up_b[ok] / n[ok, None]
    theta = -np.arcsin(np.clip(u[:, 0], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 2])
    return phi, theta
