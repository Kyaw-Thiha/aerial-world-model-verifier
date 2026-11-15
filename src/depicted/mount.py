"""Task 3: is the *published* mount the right one? Fit it and look at the residual.

With the mount published there is nothing to calibrate for the headline, so this is a check
rather than a fit: if `R_cb` were misread (transposed, or applied in the wrong direction), no
constant rotation of the camera could repair the error, and the residual against a *fitted*
rotation would stay large. If the reading is right, the fitted rotation is small and its
residual is the estimator's own bias.

The fit is per rig, on the pre-registered fit sequences only, and evaluated on the held-out
sequences of the same rig — the same discipline a fitted mount would need, applied to a
published one as a check.
"""
from __future__ import annotations

import numpy as np


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation ``D`` minimising ``sum ||D s_i - t_i||^2`` for unit vectors (Kabsch)."""
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if len(s) < 3:
        raise ValueError("need at least 3 correspondences")
    H = s.T @ t
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return D


def median_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Median angle between corresponding unit vectors, degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(c))))


def fit_mount_rotation(up_est_body: np.ndarray, up_true_body: np.ndarray,
                       iters: int = 3) -> dict:
    """Constant body-frame rotation that best maps the estimate onto the truth.

    A robust pass: fit by Kabsch, then refit on the frames whose residual is under the median
    (the heavy tail of blur failures should not drag a mount estimate), repeated a few times.
    Returns the rotation, its angle, and the residual before and after.
    """
    est = np.asarray(up_est_body, dtype=np.float64)
    tru = np.asarray(up_true_body, dtype=np.float64)
    ok = np.isfinite(est).all(axis=1) & np.isfinite(tru).all(axis=1)
    est, tru = est[ok], tru[ok]
    keep = np.ones(len(est), bool)
    D = np.eye(3)
    for _ in range(max(1, iters)):
        if keep.sum() < 3:
            break
        D = kabsch_rotation(est[keep], tru[keep])
        r = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", est @ D.T, tru), -1, 1)))
        thr = np.median(r)
        keep = r <= max(thr, 1.0)
    corrected = est @ D.T
    return {
        "n_frames": int(len(est)),
        "n_in_fit": int(keep.sum()),
        "rotation": D.tolist(),
        "rotation_angle_deg": float(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2,
                                                               -1.0, 1.0)))),
        "residual_before_deg": median_angle_deg(est, tru),
        "residual_after_deg": median_angle_deg(corrected, tru),
    }


def roll_pitch_from_up(up_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body roll and pitch (radians) implied by world up in the body frame (ZYX)."""
    u = np.asarray(up_body, dtype=np.float64).reshape(-1, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    pitch = -np.arcsin(np.clip(u[:, 0], -1.0, 1.0))
    roll = np.arctan2(u[:, 1], u[:, 2])
    return roll, pitch
