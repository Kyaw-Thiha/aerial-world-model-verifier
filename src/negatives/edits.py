"""Counterfactual attitude edits: the rotation that each negative applies to a frame.

All edits are a pure rotation about the camera centre, expressed in the world frame
``D`` and mapped into the camera frame with ``M = R_cw.T @ D @ R_cw``. The body attitude
from mocap is decomposed as ``R_bw = Rz(yaw) @ T`` with ``T = Ry(pitch) @ Rx(roll)``
(ZYX), which is what makes the three edits well defined:

* tilt scaling (``k``): ``R'_bw = Rz(yaw) @ Exp(k * log T)``. ``k = 0`` removes the tilt
  (level camera, heading kept), ``k = 1`` is the identity, ``k = 1.5`` exaggerates it.
* mirrored roll: ``R'_bw = Rz(yaw) @ Ry(pitch) @ Rx(-roll)``.
* yaw control: ``R'_bw = Rz(alpha) @ R_bw`` — a rotation about world up. It warps the
  frame by the same amount (same border loss) but cannot change the depicted tilt,
  so anything the verifier flags on it is a warp artifact.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

E3 = np.array([0.0, 0.0, 1.0])


def _axis_rotations(axis: str, angle: np.ndarray | float) -> np.ndarray:
    """Rotation matrices about a world axis for an array of angles.

    ``Rotation.from_euler`` rejects a 1-D angle array for a single-axis string in this
    scipy version, so the angle is lifted into a rotvec instead.
    """
    a = np.atleast_1d(np.asarray(angle, dtype=np.float64))
    rv = np.zeros((len(a), 3))
    rv[:, {"x": 0, "y": 1, "z": 2}[axis]] = a
    return Rotation.from_rotvec(rv).as_matrix()


def zyx_parts(R_bw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw, pitch, roll) in radians for (N, 3, 3) or (3, 3) body-to-world rotations."""
    R = np.asarray(R_bw, dtype=np.float64)
    ypr = Rotation.from_matrix(R).as_euler("ZYX")
    return ypr[..., 0], ypr[..., 1], ypr[..., 2]


def tilt_matrix(R_bw: np.ndarray) -> np.ndarray:
    """The ZYX tilt part ``Ry(pitch) @ Rx(roll)``, i.e. attitude without yaw."""
    R = np.asarray(R_bw, dtype=np.float64)
    yaw, pitch, roll = zyx_parts(R)
    return np.einsum("nij,njk->nik", _axis_rotations("y", pitch), _axis_rotations("x", roll))


def scaled_tilt_attitude(R_bw: np.ndarray, k: float) -> np.ndarray:
    """``Rz(yaw) @ T**k``: tilt scaled by ``k`` with the heading preserved.

    ``T**k`` is the matrix exponential of ``k * log T``; since ``T`` is built from two
    eigen-axis rotations the logarithm is the summed axis-angle vector and the result is
    the rotation by ``k`` times that vector. ``k = 0`` gives the identity, so the
    attitude becomes a pure yaw.
    """
    R = np.asarray(R_bw, dtype=np.float64)
    yaw, _, _ = zyx_parts(R)
    T = tilt_matrix(R)
    rv = Rotation.from_matrix(T).as_rotvec()
    Tk = Rotation.from_rotvec(k * rv).as_matrix()
    Rz = _axis_rotations("z", yaw)
    return np.einsum("nij,njk->nik", Rz, Tk) if R.ndim == 3 else Rz @ Tk


def mirrored_roll_attitude(R_bw: np.ndarray) -> np.ndarray:
    """``Rz(yaw) @ Ry(pitch) @ Rx(-roll)``."""
    R = np.asarray(R_bw, dtype=np.float64)
    yaw, pitch, roll = zyx_parts(R)
    Rz = _axis_rotations("z", yaw)
    Ry = _axis_rotations("y", pitch)
    Rx = _axis_rotations("x", -roll)
    return np.einsum("nij,njk,nkl->nil", Rz, Ry, Rx) if R.ndim == 3 else Rz @ Ry @ Rx


def delta_from_attitudes(R_new: np.ndarray, R_old: np.ndarray) -> np.ndarray:
    """World-frame rotation ``D = R_new @ R_old.T``."""
    return np.einsum("nij,nkj->nik", R_new, R_old)


def camera_frame_map(D: np.ndarray, R_cw: np.ndarray) -> np.ndarray:
    """``M = R_cw.T @ D @ R_cw``: a world rotation in camera coordinates."""
    D = np.asarray(D, dtype=np.float64)
    R = np.asarray(R_cw, dtype=np.float64)
    # M = R^T D R, i.e. M[i, l] = R[j, i] D[j, k] R[k, l]
    return np.einsum("nji,njk,nkl->nil", R, D, R)


def rotation_angle_deg(M: np.ndarray) -> np.ndarray:
    """Angle of each rotation matrix, degrees."""
    return np.degrees(Rotation.from_matrix(np.asarray(M)).magnitude())


def tilt_of_attitude_deg(R_bw: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and world up, degrees."""
    R = np.asarray(R_bw, dtype=np.float64)
    z = R[..., :, 2]
    return np.degrees(np.arccos(np.clip(z[..., 2], -1.0, 1.0)))


def yaw_control_attitude(R_bw: np.ndarray, angle_rad: np.ndarray | float) -> np.ndarray:
    """``Rz(alpha) @ R_bw``: a rotation about world up, tilt unchanged by construction."""
    R = np.asarray(R_bw, dtype=np.float64)
    a = np.asarray(angle_rad, dtype=np.float64).ravel()
    if R.ndim == 2:
        return _axis_rotations("z", float(a[0]))[0] @ R
    Rz = _axis_rotations("z", a)
    return np.einsum("nij,njk->nik", Rz, R)


def condition_maps(condition: str, R_bw: np.ndarray, R_cw: np.ndarray,
                   k: float = 1.0, yaw_rot_rad: np.ndarray | float = 0.0) -> np.ndarray:
    """Camera-frame rotations ``M`` (N, 3, 3) for one edit condition.

    ``condition`` is one of ``identity``, ``tilt_k`` (with ``k``), ``mirror_roll``,
    ``yaw_control`` (with ``yaw_rot_rad``).
    """
    if condition == "identity":
        D = np.tile(np.eye(3), (len(R_bw), 1, 1))
    elif condition == "tilt_k":
        D = delta_from_attitudes(scaled_tilt_attitude(R_bw, k), R_bw)
    elif condition == "mirror_roll":
        D = delta_from_attitudes(mirrored_roll_attitude(R_bw), R_bw)
    elif condition == "yaw_control":
        D = delta_from_attitudes(yaw_control_attitude(R_bw, yaw_rot_rad), R_bw)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return camera_frame_map(D, R_cw)
