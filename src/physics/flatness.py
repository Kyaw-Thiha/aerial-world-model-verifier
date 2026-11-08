"""Flatness: the thrust axis is fixed by the trajectory.

A quadrotor can only thrust along its body z-axis, so with world z up

    m a = T z_B - m g e3 - F_drag
    z_B = (a + g e3 + f_drag/m) / || a + g e3 + f_drag/m ||

The drag-free case is |f_drag| = 0. The linear-drag case used for the floor gate is
a single world-frame coefficient in units of 1/s:

    z_B = (a + g e3 + k_d v) / || a + g e3 + k_d v ||

The body-frame variant (Faessler et al. 2018) only rotates the drag into the
body frame; it is a diagnostic, not part of the gate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

from src.datasets.clip import QUAT_ORDERS

G0 = 9.81  # m/s^2, the g used in the flatness equation (explicit, not np defaults)


def _unit(u: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    if not np.isfinite(n).all():
        raise ValueError("non-finite norm in thrust axis")
    if (n < 1e-9).any():
        bad = int((n.ravel() < 1e-9).sum())
        raise ValueError(
            f"degenerate thrust axis for {bad} sample(s): a + g e3 (+ drag) has zero norm "
            "(free fall carries no thrust direction)"
        )
    return u / n


def implied_thrust_axis(
    a: np.ndarray,
    v: np.ndarray | None = None,
    k_d: float = 0.0,
    g: float = G0,
) -> np.ndarray:
    """Unit body z-axis implied by acceleration (and optional linear drag).

    Parameters
    ----------
    a : (N, 3) acceleration, m/s^2, world frame z-up.
    v : (N, 3) or None velocity, m/s. Required when ``k_d != 0``.
    k_d : world-frame linear drag coefficient, 1/s. Thrust must also balance the
        drag, which opposes velocity, so the term enters with a plus sign.
    g : gravitational acceleration, m/s^2.

    Returns
    -------
    (N, 3) unit vectors. Raises ValueError if any row is degenerate (free fall).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    u = a + np.array([0.0, 0.0, g])
    if k_d != 0.0:
        if v is None:
            raise ValueError("k_d != 0 requires a velocity array")
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        if len(v) != len(a):
            raise ValueError(f"a and v length mismatch: {len(a)} vs {len(v)}")
        u = u + k_d * v
    return _unit(u)


def implied_thrust_axis_body_drag(
    a: np.ndarray,
    v: np.ndarray,
    quat: np.ndarray,
    quat_order: str,
    k_xy: tuple[float, float],
    g: float = G0,
) -> np.ndarray:
    """Thrust axis with body-frame rotor drag on body x and y only.

    Drag acceleration is ``R diag(k_x, k_y, 0) R^T v`` with ``R`` body-to-world.
    The body-z coefficient is not identifiable from the thrust axis (it only
    rescales thrust), so it is fixed at zero.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    R = rotation_matrices(quat, quat_order)
    v_body = np.einsum("nji,nj->ni", R, v)  # R^T v
    d = np.array([k_xy[0], k_xy[1], 0.0])
    a_drag_world = np.einsum("nij,nj->ni", R, d * v_body)
    return _unit(a + np.array([0.0, 0.0, g]) + a_drag_world)


def rotation_matrices(quat: np.ndarray, quat_order: str) -> np.ndarray:
    """(N, 3, 3) body-to-world rotation matrices from (N, 4) quaternions."""
    if quat_order not in QUAT_ORDERS:
        raise ValueError(f"quat_order must be one of {QUAT_ORDERS}, got {quat_order!r}")
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    if quat_order == "wxyz":
        q = q[:, [1, 2, 3, 0]]
    return Rotation.from_quat(q).as_matrix()


def thrust_axis_from_quat(quat: np.ndarray, order: str) -> np.ndarray:
    """True body z-axis in the world frame, from body-to-world attitude.

    ``order`` is ``"xyzw"`` or ``"wxyz"``. The returned (N, 3) rows are unit
    vectors because a rotation matrix is orthonormal.
    """
    R = rotation_matrices(quat, order)
    return R[:, :, 2]


def tilt_deg(z_B: np.ndarray, g: float = G0) -> np.ndarray:
    """Tilt of a thrust axis away from world up, in degrees.

    Accepts (..., 3) unit vectors. Uses the z-component directly so a nadir
    pointing thrust axis (z = -1) returns 180 degrees rather than failing.
    """
    z = np.asarray(z_B, dtype=np.float64)
    if z.shape[-1] != 3:
        raise ValueError(f"expected last axis of length 3, got {z.shape}")
    c = np.clip(z[..., 2], -1.0, 1.0)
    return np.degrees(np.arccos(c))


def tilt_from_roll_pitch(roll: np.ndarray, pitch: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Tilt of the body z-axis for a ZYX attitude, in degrees.

    Yaw does not tilt the thrust axis, so only roll and pitch enter. Used to turn a
    recorded attitude into the same quantity the flatness equation predicts, so a
    dataset's *depicted* tilt and its *implied* tilt can be compared without two
    copies of this formula drifting apart.
    """
    r = np.radians(roll) if degrees else np.asarray(roll, dtype=np.float64)
    p = np.radians(pitch) if degrees else np.asarray(pitch, dtype=np.float64)
    c = np.clip(np.cos(r) * np.cos(p), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def implied_tilt_from_positions(
    pos: np.ndarray,
    dt: float,
    window: int = 11,
    polyorder: int = 3,
    k_d: float = 0.0,
    g: float = G0,
) -> tuple[np.ndarray, np.ndarray]:
    """Implied tilt (deg) from a position track, plus a mask of usable samples.

    One implementation shared by the mocap floor, the ANWM audit and later pixel
    stages: differentiate twice, invert flatness, take the tilt. Invalid samples
    (window edges, free fall) are NaN in ``tilt`` and False in ``mask``.
    """
    from src.physics.diff import savgol_derivatives

    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    w = min(window, n - 1)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        raise ValueError(f"too few frames ({n}) for window {w} and polyorder {polyorder}")
    vel, acc = savgol_derivatives(pos, dt, w, polyorder)
    half = w // 2
    sl = slice(half, n - half)
    ok = valid_thrust_mask(acc[sl], vel[sl], k_d, g)
    tilt = np.full(n, np.nan)
    idx = np.arange(sl.start, sl.stop)[ok]
    if len(idx):
        tilt[idx] = tilt_deg(implied_thrust_axis(acc[sl][ok], vel[sl][ok], k_d, g))
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return tilt, mask


def angle_between_deg(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Angle between two vectors in degrees, robust near 0 and 180 degrees.

    ``arccos`` loses precision as the angle goes to zero because the dot product
    saturates at 1; ``arctan2(||u x w||, u . w)`` does not.
    """
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if u.shape != w.shape:
        raise ValueError(f"shape mismatch: {u.shape} vs {w.shape}")
    cross = np.linalg.norm(np.cross(u, w), axis=-1)
    dot = np.sum(u * w, axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def valid_thrust_mask(
    a: np.ndarray, v: np.ndarray | None = None, k_d: float = 0.0, g: float = G0,
    min_norm: float = 1.0,
) -> np.ndarray:
    """Boolean mask of samples whose thrust axis is well defined and well conditioned."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    u = a + np.array([0.0, 0.0, g])
    if k_d != 0.0:
        if v is None:
            raise ValueError("k_d != 0 requires a velocity array")
        u = u + k_d * np.asarray(v, dtype=np.float64).reshape(-1, 3)
    return np.linalg.norm(u, axis=1) >= min_norm


def _median_residual_deg(acc, vel, true_axis, k_d: float, g: float, mask) -> float:
    r = angle_between_deg(implied_thrust_axis(acc[mask], vel[mask], k_d, g), true_axis[mask])
    return float(np.median(r))


def fit_kd(
    acc: np.ndarray,
    vel: np.ndarray,
    true_axis: np.ndarray,
    g: float = G0,
    bounds: tuple[float, float] = (0.0, 3.0),
    mask: np.ndarray | None = None,
    xatol: float = 1e-4,
) -> dict:
    """Fit the scalar world-frame drag coefficient by minimising median residual.

    Median (not mean) so a handful of mocap dropouts or aggressive transients
    cannot drag the fit. Returns the estimate plus the loss before and after.
    """
    acc = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    true_axis = np.asarray(true_axis, dtype=np.float64).reshape(-1, 3)
    if mask is None:
        mask = valid_thrust_mask(acc, vel, 0.0, g)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 10:
        raise ValueError(f"fit_kd needs >= 10 valid samples, got {int(mask.sum())}")
    res = minimize_scalar(
        lambda k: _median_residual_deg(acc, vel, true_axis, float(k), g, mask),
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol},
    )
    k = float(res.x)
    return {
        "k_d": k,
        "median_deg_free": _median_residual_deg(acc, vel, true_axis, 0.0, g, mask),
        "median_deg_fit": _median_residual_deg(acc, vel, true_axis, k, g, mask),
        "n_valid": int(mask.sum()),
        "bounds": list(bounds),
        "loss": float(res.fun),
    }


def fit_body_drag(
    acc: np.ndarray,
    vel: np.ndarray,
    true_axis: np.ndarray,
    quat: np.ndarray,
    quat_order: str,
    g: float = G0,
    bounds: tuple[float, float] = (0.0, 3.0),
    mask: np.ndarray | None = None,
) -> dict:
    """Fit body-frame x/y drag coefficients (diagnostic only).

    Nelder-Mead on median residual, started from the scalar-drag solution. The
    extra parameter is only worth its degrees of freedom if the high-speed
    residual actually drops, which is what the experiment reports.
    """
    acc = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    true_axis = np.asarray(true_axis, dtype=np.float64).reshape(-1, 3)
    if mask is None:
        mask = valid_thrust_mask(acc, vel, 0.0, g)
    mask = np.asarray(mask, dtype=bool)

    def loss(k: np.ndarray) -> float:
        kx, ky = float(abs(k[0])), float(abs(k[1]))
        z = implied_thrust_axis_body_drag(acc[mask], vel[mask], quat[mask], quat_order,
                                         (kx, ky), g)
        return float(np.median(angle_between_deg(z, true_axis[mask])))

    x0 = np.array([0.1, 0.1])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
    kx, ky = abs(float(res.x[0])), abs(float(res.x[1]))
    return {
        "k_x": kx, "k_y": ky,
        "median_deg": float(res.fun),
        "median_deg_free": loss(np.array([0.0, 0.0])),
        "n_valid": int(mask.sum()),
        "bounds": list(bounds),
        "converged": bool(res.success),
    }
