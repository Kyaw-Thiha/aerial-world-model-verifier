"""Attitude injection for ANWM: thrust axis -> body rotation -> camera pose for FFP.

What the release does (verified from source and data, see reports/b05_*):

* `pose` is **camera-to-world**: `anwm/projection.py::reproject_depth_to_other_pose` computes
  `points_world = pose_src @ points_cam` and then `points_dst = inv(pose_dst) @ points_world`.
* the preprocessing notebook builds `pose = create_pose_matrix(xyz, quaternion) @ t_ec`, with
  `t_ec = [[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]` (rotation only, no translation), so
  `R_wc = R_wb @ R_ec` and the optical axis (camera +z) maps to body +x (forward).
* `R_ec` recovered from the released poses equals `t_ec[:3,:3]` to 0 machine epsilon (spread
  2.2e-16 over 269 frames), and every released pitch and roll is exactly 0.
* `anwm.projection.euler_to_quaternion(yaw, pitch, roll)` is
  `scipy Rotation.from_euler("zyx", [...])`, which is **extrinsic** z-y-x, i.e.
  `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)` -- *not* the intrinsic `Rz Ry Rx` that the usual
  "ZYX yaw-pitch-roll" wording implies. A tilted matrix is still expressible in that
  parameterisation (any rotation is), but the three angles are not (heading, pitch, roll) of the
  vehicle. We therefore build the pose matrix directly from the attitude we want, and report the
  equivalent release triple only for completeness.

Convention used for the injection (intrinsic ZYX, yaw then pitch then roll):

    R_wb = Rz(psi) @ Ry(theta) @ Rx(phi)
    z_B  = R_wb[:, 2]                     (body up = thrust axis, world frame)
    R_wc = R_wb @ R_ec                    (camera pose fed to FFP)

Given a desired unit thrust axis `z_B` and the camera yaw `psi`:

    z'   = Rz(-psi) @ z_B
    phi   = -asin(z'_y)
    theta = atan2(z'_x, z'_z)

which is exact for `cos(phi) > 0`, i.e. for every attitude with |roll| < 90 deg.
"""
from __future__ import annotations

import numpy as np

from src.physics.flatness import G0, implied_thrust_axis


def _as_rot(yaw, pitch, roll):
    yaw = np.atleast_1d(np.asarray(yaw, dtype=np.float64))
    pitch = np.atleast_1d(np.asarray(pitch, dtype=np.float64))
    roll = np.atleast_1d(np.asarray(roll, dtype=np.float64))
    n = max(len(yaw), len(pitch), len(roll))
    yaw, pitch, roll = (np.broadcast_to(v, (n,)) for v in (yaw, pitch, roll))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = cy; Rz[:, 0, 1] = -sy
    Rz[:, 1, 0] = sy; Rz[:, 1, 1] = cy; Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = cp; Ry[:, 0, 2] = sp
    Ry[:, 2, 0] = -sp; Ry[:, 2, 2] = cp; Ry[:, 1, 1] = 1.0
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1.0; Rx[:, 1, 1] = cr
    Rx[:, 1, 2] = -sr; Rx[:, 2, 1] = sr; Rx[:, 2, 2] = cr
    return Rz @ Ry @ Rx


def body_rotation(yaw, pitch, roll) -> np.ndarray:
    """(N,3,3) body-to-world rotation, intrinsic ZYX (Rz @ Ry @ Rx)."""
    return _as_rot(yaw, pitch, roll)


def thrust_from_accel(accel: np.ndarray, k_d: float = 0.0, vel: np.ndarray | None = None,
                      g: float = G0) -> np.ndarray:
    """Unit thrust axis implied by acceleration (flatness), (N,3)."""
    return implied_thrust_axis(accel, vel, k_d, g)


def roll_pitch_from_thrust(z_B: np.ndarray, yaw) -> tuple[np.ndarray, np.ndarray]:
    """(phi, theta) whose body z-axis is `z_B`, for the given heading."""
    z = np.asarray(z_B, dtype=np.float64).reshape(-1, 3)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    yaw = np.atleast_1d(np.asarray(yaw, dtype=np.float64))
    if len(yaw) == 1:
        yaw = np.broadcast_to(yaw, (len(z),))
    c, s = np.cos(yaw), np.sin(yaw)
    zx = c * z[:, 0] + s * z[:, 1]
    zy = -s * z[:, 0] + c * z[:, 1]
    zz = z[:, 2]
    if np.any(np.abs(zy) >= 1.0):
        raise ValueError("thrust axis has |horizontal| >= 1: vertical, no real attitude")
    phi = -np.arcsin(zy)
    theta = np.arctan2(zx, zz)
    return phi, theta


def pose_with_attitude(pos: np.ndarray, yaw: np.ndarray, z_B: np.ndarray,
                       R_ec: np.ndarray) -> np.ndarray:
    """(N,4,4) camera-to-world poses whose body z-axis is `z_B` and heading is `yaw`."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    phi, theta = roll_pitch_from_thrust(z_B, yaw)
    R_wb = body_rotation(yaw, theta, phi)
    poses = np.tile(np.eye(4), (len(pos), 1, 1))
    poses[:, :3, :3] = R_wb @ np.asarray(R_ec, dtype=np.float64)[None]
    poses[:, :3, 3] = pos
    return poses


def body_z_from_poses(poses: np.ndarray, R_ec: np.ndarray) -> np.ndarray:
    """World body z-axis recovered from camera-to-world poses (inverse of the construction)."""
    R_wc = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    R_wb = R_wc @ np.asarray(R_ec, dtype=np.float64).T
    return R_wb[:, :, 2]


def release_euler_triple(R_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw, pitch, roll) as ANWM's `euler_to_quaternion` would take them (extrinsic z-y-x).

    Reported for completeness and checked by reconstruction; the injected pose is built from the
    matrix, not from this triple, because the triple is not (heading, pitch, roll) once tilted.
    """
    from scipy.spatial.transform import Rotation

    eul = Rotation.from_matrix(np.asarray(R_wb, dtype=np.float64)).as_euler("zyx")
    return eul[:, 0], eul[:, 1], eul[:, 2]


def rotation_homography(K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """K R K^-1 for a rotation R expressed from source-camera to target-camera coordinates."""
    K = np.asarray(K, dtype=np.float64)
    return K @ np.asarray(R, dtype=np.float64) @ np.linalg.inv(K)
