"""Lean geometry: required lean from the commanded path, depicted lean from gravity.

Conventions (identical to the rest of the project, see ``src/perception/frames.py``):

* world z-up, SI units; body FLU (x forward, y left, z up); yaw about world z.
* ZYX attitude ``R_bw = Rz(yaw) Ry(pitch) Rx(roll)``.
* The lean vector is the **horizontal part of the unit thrust axis in the yaw-aligned
  frame**: ``l = (cos(phi) sin(theta), -sin(phi))``.  It is not an angle in radians but a
  dimensionless horizontal component (``sin`` of the tilt); at the small angles of this
  data the two differ by less than 0.1 % below 5 deg.

Both sides are expressed in the same frame, so a regression of depicted on required lean
has a slope of 1 when the pixels agree with the path, 0 when tilt is removed, and -1 on
the roll axis when roll is mirrored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from src.physics.flatness import G0

E3 = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ frame helpers
def yaw_to_frame(yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    return np.cos(yaw), np.sin(yaw)


def world_to_yaw_frame(vec: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate the horizontal part of ``vec`` (N, 2 or N, 3) by ``Rz(-yaw)``."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 2 or vec.shape[1] < 2:
        raise ValueError(f"vec must be (N, >=2), got {vec.shape}")
    c, s = yaw_to_frame(yaw)
    if len(vec) != len(c):
        raise ValueError(f"length mismatch: {len(vec)} vectors vs {len(c)} yaws")
    x, y = vec[:, 0], vec[:, 1]
    return np.column_stack([c * x + s * y, -s * x + c * y])


def lean_to_roll_pitch(lean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``l = (cos(phi) sin(theta), -sin(phi))`` -> (roll phi, pitch theta), radians."""
    lean = np.asarray(lean, dtype=np.float64)
    if lean.ndim != 2 or lean.shape[1] != 2:
        raise ValueError(f"lean must be (N, 2), got {lean.shape}")
    roll = -np.arcsin(np.clip(lean[:, 1], -1.0, 1.0))
    cos_roll = np.cos(roll)
    # lean_x can leave [-1, 1] only through a numerically bad input; clip and report.
    pitch = np.arcsin(np.clip(lean[:, 0] / np.where(np.abs(cos_roll) > 1e-9,
                                                    cos_roll, np.nan), -1.0, 1.0))
    return roll, pitch


def roll_pitch_to_lean(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    roll = np.asarray(roll, dtype=np.float64).ravel()
    pitch = np.asarray(pitch, dtype=np.float64).ravel()
    if len(roll) != len(pitch):
        raise ValueError(f"length mismatch: {len(roll)} vs {len(pitch)}")
    return np.column_stack([np.cos(roll) * np.sin(pitch), -np.sin(roll)])


def depicted_lean_from_gravity(up_cam: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """Depicted lean in the yaw-aligned frame from world up expressed in the camera.

    ``up_cam`` (N, 3) is the unit world-up direction in camera coordinates (GeoCalib's
    convention); ``R_cb`` (3, 3) maps camera -> body.  Then ``u_b = R_cb @ up_cam`` gives
    the body-frame world up, and for a ZYX attitude

        u_b = (-sin(theta), sin(phi) cos(theta), cos(phi) cos(theta))

    so ``theta = -asin(u_b_x)`` and ``phi = atan2(u_b_y, u_b_z)``.  The yaw does not enter:
    roll and pitch are observable from gravity alone, and the lean is already yaw-free.
    """
    up_cam = np.asarray(up_cam, dtype=np.float64)
    if up_cam.ndim != 2 or up_cam.shape[1] != 3:
        raise ValueError(f"up_cam must be (N, 3), got {up_cam.shape}")
    R_cb = np.asarray(R_cb, dtype=np.float64)
    if R_cb.shape != (3, 3):
        raise ValueError(f"R_cb must be (3, 3), got {R_cb.shape}")
    norm = np.linalg.norm(up_cam, axis=1, keepdims=True)
    good = np.isfinite(up_cam).all(axis=1) & (norm[:, 0] > 0)
    unit = np.full_like(up_cam, np.nan)
    unit[good] = up_cam[good] / norm[good]
    u_b = unit @ R_cb.T
    pitch = -np.arcsin(np.clip(u_b[:, 0], -1.0, 1.0))
    roll = np.arctan2(u_b[:, 1], u_b[:, 2])
    lean = roll_pitch_to_lean(roll, pitch)
    lean[~good] = np.nan
    return lean


# ------------------------------------------------------------------ path side
@dataclass
class RequiredLean:
    """Per-frame required lean and its regressors, all in the yaw-aligned frame."""

    l_req0: np.ndarray        # (N, 2) drag-free required lean
    velocity_reg: np.ndarray  # (N, 2) horizontal velocity / g, units s
    tilt_deg: np.ndarray      # (N,) required tilt angle from the unit thrust axis
    vel: np.ndarray           # (N, 3) world velocity, m/s
    acc: np.ndarray           # (N, 3) world acceleration, m/s^2
    mask: np.ndarray          # (N,) True where the SG stencil is interior and finite
    window: int               # samples
    dt_s: float


def _savgol_window(n: int, dt_s: float, window_s: float, polyorder: int) -> int:
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError(f"dt_s must be positive and finite, got {dt_s!r}")
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s!r}")
    w = int(round(window_s / dt_s))
    if w % 2 == 0:
        w += 1
    w = max(polyorder + 2, w)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w


def required_lean(
    pos: np.ndarray,
    yaw: np.ndarray,
    t: np.ndarray,
    window_s: float = 0.37,
    polyorder: int = 3,
    g: float = G0,
) -> RequiredLean:
    """Required lean from the commanded path.

    Velocity and acceleration come from Savitzky-Golay differentiation with a window
    given in **seconds**; the sample interval is the median of ``diff(t)`` (the TII frame
    grid is close to uniform at 12 Hz, with occasional dropped frames).  Frames whose
    stencil is one-sided are masked, because a one-sided second derivative is biased.
    """
    pos = np.asarray(pos, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N, 3), got {pos.shape}")
    n = len(t)
    if len(pos) != n or len(yaw) != n:
        raise ValueError(f"length mismatch: t {n}, pos {len(pos)}, yaw {len(yaw)}")
    if n < 7:
        raise ValueError(f"need at least 7 frames for a second derivative, got {n}")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    dt_s = float(np.median(np.diff(t)))
    w = _savgol_window(n, dt_s, window_s, polyorder)

    vel = savgol_filter(pos, w, polyorder, deriv=1, delta=dt_s, axis=0, mode="interp")
    acc = savgol_filter(pos, w, polyorder, deriv=2, delta=dt_s, axis=0, mode="interp")

    q = acc + g * E3
    norm = np.linalg.norm(q, axis=1)
    u0 = np.full_like(q, np.nan)
    good = np.isfinite(norm) & (norm > 1e-9)
    u0[good] = q[good] / norm[good, None]
    l_req0 = world_to_yaw_frame(u0[:, :2], yaw)
    velocity_reg = world_to_yaw_frame(vel[:, :2] / g, yaw)
    tilt_deg = np.degrees(np.arccos(np.clip(u0[:, 2], -1.0, 1.0)))

    half = w // 2
    mask = np.zeros(n, dtype=bool)
    mask[half:n - half] = True
    mask &= np.isfinite(l_req0).all(axis=1) & np.isfinite(velocity_reg).all(axis=1)
    mask &= np.isfinite(pos).all(axis=1)
    return RequiredLean(l_req0=l_req0, velocity_reg=velocity_reg, tilt_deg=tilt_deg,
                        vel=vel, acc=acc, mask=mask, window=w, dt_s=dt_s)


def principal_spread_deg(l_req0: np.ndarray) -> float:
    """Standard deviation of the required lean along its principal axis, in degrees."""
    x = np.asarray(l_req0, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3:
        return 0.0
    c = x - x.mean(axis=0)
    cov = c.T @ c / len(c)
    ev = np.linalg.eigvalsh(cov)
    return float(np.degrees(np.sqrt(max(ev[-1], 0.0))))
