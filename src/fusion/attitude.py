"""Fusion of a monocular VO trajectory with a per-frame gravity estimate.

Two pieces, both cheap and testable:

* :func:`gravity_align` — one rotation per clip that takes the robust mean of the
  per-frame gravity measurement expressed in the VO world frame onto world up.
  The spread of those per-frame directions is returned; a large spread means the
  gravity estimates disagree and the clip should not be trusted.
* :func:`complementary_filter` — propagate attitude with the VO relative rotation
  and correct toward each measured gravity direction with a fixed gain. The gain
  and the update rate are explicit arguments; a gain tuned on one flight is not a
  universal constant and is recorded per run.

Conventions follow :mod:`src.perception.frames`: camera RDF, body FLU, world z up,
``up`` is the direction away from gravity, and ``R_cb`` maps camera->body.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def up_in_vo_frame(R_vo: np.ndarray, up_cam: np.ndarray) -> np.ndarray:
    """Gravity direction per frame expressed in the VO world frame.

    ``R_vo`` maps camera->VO-world (columns are camera axes in the VO frame).
    """
    u = np.einsum("nij,nj->ni", R_vo, up_cam)
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.where(n > 0, n, 1.0)


def gravity_align(R_vo: np.ndarray, up_cam: np.ndarray, weights: np.ndarray | None = None
                  ) -> dict:
    """One rotation taking the mean measured up (in VO frame) to world up."""
    uv = up_in_vo_frame(R_vo, up_cam)
    w = np.ones(len(uv)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mean = (uv * w[:, None]).sum(axis=0)
    mean /= np.linalg.norm(mean)
    R_align = rotation_between(mean, np.array([0.0, 0.0, 1.0]))
    angles = np.degrees(np.arccos(np.clip(uv @ mean, -1.0, 1.0)))
    return {"R_align": R_align, "mean_up_vo": mean,
            "spread_median_deg": float(np.median(angles)),
            "spread_p90_deg": float(np.percentile(angles, 90)),
            "spread_n": int(len(angles))}


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    Kx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + Kx + Kx @ Kx * ((1.0 - c) / s**2)


def relative_rotations(R: np.ndarray) -> np.ndarray:
    """``R[t+1] @ R[t]^T`` — the rotation carrying frame t to frame t+1."""
    return np.einsum("nij,nkj->nik", R[1:], R[:-1])


@dataclass
class FilterResult:
    R: np.ndarray                  # (N, 3, 3) camera->aligned world, filtered
    up_err_deg: np.ndarray         # (N,) up-vs-world-up angle *before* correction
    up_err_after_deg: np.ndarray   # (N,) the same *after* correction
    gain: float


def complementary_filter(R_vo_rel: np.ndarray, up_cam: np.ndarray, gain: float,
                         R0: np.ndarray | None = None, R_align: np.ndarray | None = None
                         ) -> FilterResult:
    """Complementary filter: VO rotation predicts, gravity measurement corrects.

    Parameters
    ----------
    R_vo_rel : (N-1, 3, 3) rotations carrying camera frame t to t+1 (VO).
    up_cam : (N, 3) measured up direction per frame in camera coordinates.
    gain : correction fraction per frame in [0, 1]. Applied as a slerp of the
        predicted up direction toward world up.
    R0 : initial attitude (camera->aligned world). Default: the rotation implied
        by the first gravity measurement.
    R_align : rotation from the VO world frame to the aligned world frame, if the
        caller wants the output in the gravity-aligned VO frame.
    """
    if not 0.0 <= gain <= 1.0:
        raise ValueError(f"gain must be in [0, 1], got {gain}")
    N = len(up_cam)
    if len(R_vo_rel) != N - 1:
        raise ValueError(f"need {N - 1} relative rotations, got {len(R_vo_rel)}")
    up_cam = up_cam / np.linalg.norm(up_cam, axis=1, keepdims=True)

    if R0 is None:
        R0 = rotation_between(up_cam[0], np.array([0.0, 0.0, 1.0]))
    R = np.empty((N, 3, 3))
    err = np.empty(N)
    err_after = np.empty(N)
    R_cur = np.array(R0, dtype=np.float64)
    for t in range(N):
        if t > 0:
            R_cur = R_vo_rel[t - 1] @ R_cur
        up_pred = R_cur @ up_cam[t]
        up_pred = up_pred / np.linalg.norm(up_pred)
        err[t] = np.degrees(np.arccos(np.clip(up_pred[2], -1.0, 1.0)))
        R_corr = rotation_between(up_pred, np.array([0.0, 0.0, 1.0]))
        ang = np.arccos(np.clip((np.trace(R_corr) - 1.0) / 2.0, -1.0, 1.0))
        if ang > 1e-12:
            R_step = Rotation.from_rotvec(Rotation.from_matrix(R_corr).as_rotvec()
                                          * gain).as_matrix()
            R_cur = R_step @ R_cur
        else:
            R_cur = R_corr @ R_cur
        up_now = R_cur @ up_cam[t]
        up_now = up_now / np.linalg.norm(up_now)
        err_after[t] = np.degrees(np.arccos(np.clip(up_now[2], -1.0, 1.0)))
        R[t] = R_cur
    if R_align is not None:
        R = np.einsum("ij,njk->nik", R_align, R)
    return FilterResult(R=R, up_err_deg=err, up_err_after_deg=err_after,
                        gain=float(gain))


def body_z_from_camera(R_cam_to_world: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """Body z axis (thrust axis) in the aligned world frame, per frame.

    ``R_cb`` maps camera->body, so body z in camera coordinates is its third row.
    """
    z_cam = np.asarray(R_cb, dtype=np.float64)[2, :]
    return np.einsum("nij,j->ni", R_cam_to_world, z_cam)


def lean_direction_deg(z_world: np.ndarray) -> np.ndarray:
    """Azimuth of the horizontal part of a thrust axis, radians in [-pi, pi)."""
    xy = z_world[:, :2]
    return np.arctan2(xy[:, 1], xy[:, 0])


def lean_error_deg(est: np.ndarray, tru: np.ndarray, min_tilt_deg: float = 10.0,
                   yaw_align: float | None = None) -> dict:
    """Angle between estimated and true horizontal lean directions.

    ``yaw_align`` is one constant azimuth offset for the clip (evaluation only,
    because the VO frame's azimuth is arbitrary). If ``None`` it is estimated as
    the circular mean of the difference on the evaluated frames.
    """
    e = lean_direction_deg(est)
    t = lean_direction_deg(tru)
    tilt_est = np.degrees(np.arccos(np.clip(est[:, 2], -1, 1)))
    tilt_tru = np.degrees(np.arccos(np.clip(tru[:, 2], -1, 1)))
    mask = (tilt_tru > min_tilt_deg) & (tilt_est > 1e-6)
    if mask.sum() == 0:
        return {"n": 0}
    d = np.angle(np.exp(1j * (e[mask] - t[mask])))
    if yaw_align is None:
        yaw_align = float(np.angle(np.mean(np.exp(1j * d))))
    d = np.angle(np.exp(1j * (d - yaw_align)))
    return {"n": int(mask.sum()), "yaw_align_rad": float(yaw_align),
            "median_abs_deg": float(np.degrees(np.median(np.abs(d)))),
            "p90_abs_deg": float(np.degrees(np.percentile(np.abs(d), 90)))}
