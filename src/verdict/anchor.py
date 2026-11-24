"""Path anchors: a quiet window in the trajectory pins the start attitude by physics.

Differential flatness says the thrust axis is ``q/|q|`` with ``q = a + g e3 + k_d v``. The lean is
zero exactly when ``q`` is vertical, so a window where the *path* implies a small thrust-axis tilt
also implies that the world-up direction in the camera is the mount rotation applied to vertical:

    u_cam(t_a) = R_cb^T e3.

Nothing here reads a pixel. The window is chosen from the trajectory alone, and the resulting start
attitude is propagated with the existing Track N machinery
(``up_k = (R_k^T R_a) u0`` -> :func:`src.verdict.lean.depicted_lean_from_gravity`).

Two structural facts this module makes explicit (both are used by the N3 reports):

* the anchor error -- the angle between the true world-up in the camera and ``R_cb^T e3`` -- equals
  the clip's true total tilt at the anchor frame, ``angle(R_bw^T e3, e3)``. It does not contain the
  mount at all: a mount error rotates the anchor direction, it does not bias this angle.
* a mount error is therefore not a constant lean offset either. It cancels at the anchor frame and
  grows with the rotation between the anchor and the frame being read (see ``mount_rotated``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])

# the values the N07/N2A required side uses (src.physics.flatness.G0 is 9.81 and is not used here)
G_ANCHOR = 9.80665
K_D_ANCHOR = 0.295


def implied_thrust_axis(acc: np.ndarray, vel: np.ndarray, g: float = G_ANCHOR,
                        k_d: float = K_D_ANCHOR) -> np.ndarray:
    """Unit thrust axis ``(a + g e3 + k_d v)/|.|``; NaN where the norm is degenerate."""
    acc = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    if len(acc) != len(vel):
        raise ValueError(f"acc/vel length mismatch: {len(acc)} vs {len(vel)}")
    q = acc + g * E3 + k_d * vel
    n = np.linalg.norm(q, axis=1)
    out = np.full_like(q, np.nan)
    good = np.isfinite(n) & (n > 1e-9)
    out[good] = q[good] / n[good, None]
    return out


def implied_tilt_deg(acc: np.ndarray, vel: np.ndarray, g: float = G_ANCHOR,
                     k_d: float = K_D_ANCHOR) -> np.ndarray:
    """Angle in degrees between the implied thrust axis and world vertical (NaN where undefined)."""
    u = implied_thrust_axis(acc, vel, g=g, k_d=k_d)
    return np.degrees(np.arccos(np.clip(u[:, 2], -1.0, 1.0)))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between two stacks of unit vectors (NaN propagates)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


@dataclass(frozen=True)
class AnchorWindow:
    """One qualifying quiet window, indices into the truth grid."""

    start: int
    stop: int  # exclusive
    mean_tilt_deg: float
    argmin: int
    t_argmin_s: float


def find_windows(theta_deg: np.ndarray, t: np.ndarray, thr_deg: float, min_dur_s: float,
                 valid: np.ndarray | None = None) -> list[AnchorWindow]:
    """Maximal runs with ``theta <= thr`` lasting at least ``min_dur_s``.

    ``valid`` masks frames that may not be used (NaN samples are always invalid). The duration is
    measured as ``t[stop-1] - t[start]`` so a window never spans a gap.
    """
    theta_deg = np.asarray(theta_deg, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if len(theta_deg) != len(t):
        raise ValueError(f"theta/t length mismatch: {len(theta_deg)} vs {len(t)}")
    good = np.isfinite(theta_deg) & np.isfinite(t)
    if valid is not None:
        good &= np.asarray(valid, bool).ravel()
    below = good & (theta_deg <= thr_deg)
    windows: list[AnchorWindow] = []
    i, n = 0, len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        start, stop = i, j + 1
        if t[stop - 1] - t[start] >= min_dur_s:
            seg = theta_deg[start:stop]
            arg = start + int(np.argmin(seg))
            windows.append(AnchorWindow(start=start, stop=stop,
                                        mean_tilt_deg=float(np.mean(seg)),
                                        argmin=arg, t_argmin_s=float(t[arg])))
        i = stop
    return windows


def select_anchor(windows: list[AnchorWindow]) -> AnchorWindow | None:
    """The window with the smallest mean implied tilt; ties go to the earliest."""
    if not windows:
        return None
    best = min(windows, key=lambda w: (round(w.mean_tilt_deg, 9), w.t_argmin_s))
    return best


def select_anchor_first(windows: list[AnchorWindow]) -> AnchorWindow | None:
    """The earliest qualifying window (a system takes the first hover it gets)."""
    return min(windows, key=lambda w: w.t_argmin_s) if windows else None


def first_sustained_motion(t: np.ndarray, speed: np.ndarray, thr: float = 2.0,
                          dur_s: float = 0.5) -> float:
    """First time the path exceeds ``thr`` and stays above it for ``dur_s``; +inf if never."""
    t = np.asarray(t, float)
    above = np.asarray(speed, float) >= thr
    if not above.any():
        return float("inf")
    n_need = max(int(round(dur_s / float(np.median(np.diff(t))))), 1)
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= n_need:
            return float(t[i - n_need + 1])
    return float("inf")


def last_sustained_motion(t: np.ndarray, speed: np.ndarray, thr: float = 2.0,
                          dur_s: float = 0.5) -> float:
    """Last time the path is above ``thr`` for ``dur_s``; -inf if never."""
    t = np.asarray(t, float)
    above = np.asarray(speed, float) >= thr
    if not above.any():
        return float("-inf")
    n_need = max(int(round(dur_s / float(np.median(np.diff(t))))), 1)
    run = 0
    for i in range(len(above) - 1, -1, -1):
        run = run + 1 if above[i] else 0
        if run >= n_need:
            return float(t[i])
    return float("-inf")


def snap_to_grid(t_s: float, t_grid: np.ndarray) -> int:
    """Index of the grid sample nearest ``t_s`` (ties to the earlier sample)."""
    t_grid = np.asarray(t_grid, dtype=np.float64).ravel()
    if not np.isfinite(t_s):
        raise ValueError("cannot snap a non-finite time")
    return int(np.argmin(np.abs(t_grid - t_s)))


def up_cam_at_anchor(R_cb: np.ndarray) -> np.ndarray:
    """World-up in the camera frame implied by a zero lean: ``R_cb^T e3``."""
    return np.asarray(R_cb, dtype=np.float64).T @ E3


def mount_rotated(R_cb: np.ndarray, deg: float, axis: str = "x") -> np.ndarray:
    """``R_cb`` with the camera pitched ``deg`` further about the body ``axis`` ('' or 'y')."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    if axis == "x":
        D = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == "y":
        D = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    elif axis == "z":
        D = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return np.asarray(R_cb, dtype=np.float64) @ D


def propagate_from(R: np.ndarray, idx_anchor: int, u0: np.ndarray,
                   idx_eval: np.ndarray) -> np.ndarray:
    """``up_k = (R_k^T R_a) u0`` at the frames ``idx_eval``; NaN where a frame is missing.

    ``R`` is the (n, 3, 3) camera-to-world rotation sequence. Works in both directions in time:
    nothing in the formula assumes ``idx_eval >= idx_anchor``.
    """
    R = np.asarray(R, dtype=np.float64)
    idx_eval = np.asarray(idx_eval, dtype=np.int64)
    out = np.full((len(idx_eval), 3), np.nan)
    if idx_anchor < 0 or idx_anchor >= len(R):
        return out
    good = (idx_eval >= 0) & (idx_eval < len(R))
    if not good.any():
        return out
    Ra = R[idx_anchor]
    rel = np.einsum("nij->nji", R[idx_eval[good]]) @ Ra  # R_k^T R_a
    out[good] = rel @ np.asarray(u0, dtype=np.float64).reshape(3)
    return out


def propagate_from_nearest(R: np.ndarray, anchors: list[int], u0: np.ndarray,
                           idx_eval: np.ndarray) -> np.ndarray:
    """Per frame, propagate from the nearest anchor frame (multi-anchor arm).

    Ties (a frame exactly between two anchors) go to the earlier anchor.
    """
    R = np.asarray(R, dtype=np.float64)
    idx_eval = np.asarray(idx_eval, dtype=np.int64)
    out = np.full((len(idx_eval), 3), np.nan)
    if not anchors or len(R) == 0:
        return out
    a = np.asarray(sorted(set(int(x) for x in anchors)), dtype=np.int64)
    good = (idx_eval >= 0) & (idx_eval < len(R))
    if not good.any():
        return out
    rows = np.flatnonzero(good)
    idx = idx_eval[rows]
    p = np.clip(np.searchsorted(a, idx), 0, len(a) - 1)
    cand = a[p]
    prev = a[np.clip(p - 1, 0, len(a) - 1)]
    pick = np.where(np.abs(idx - prev) < np.abs(cand - idx), prev, cand)
    u0 = np.asarray(u0, dtype=np.float64).reshape(3)
    for row, anchor in zip(rows, pick):
        out[row] = (R[idx_eval[row]].T @ R[anchor]) @ u0
    return out
