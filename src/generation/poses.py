"""Camera poses in ANWM's released convention, and how to build new ones.

The release stores, per frame, `pose[i] = [R_z(yaw_i) @ R_ec | xyz_i]` (preprocessing notebook
cell 14), where `R_ec` is the fixed world->camera axis change (FLU body -> RDF camera). FFP
reprojects context depth with these matrices, so a new target pose must be built in exactly the
same convention or the projected prior is silently rotated. `camera_axis_change` recovers `R_ec`
from released data and `pose_from_position_yaw` builds new poses with it.
"""
from __future__ import annotations

import numpy as np


def yaw_matrix(yaw: float | np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation about world z."""
    y = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(y), np.sin(y)
    R = np.zeros(y.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1] = c, -s
    R[..., 1, 0], R[..., 1, 1] = s, c
    R[..., 2, 2] = 1.0
    return R


def camera_axis_change(poses: np.ndarray, yaws: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, float]:
    """Recover `R_ec` from released poses and check it is the same in every frame."""
    R = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    yaw = np.asarray(yaws, dtype=np.float64).ravel()
    if len(R) != len(yaw):
        raise ValueError(f"poses/yaws length mismatch: {len(R)} vs {len(yaw)}")
    out = np.einsum("nij,njk->nik", yaw_matrix(yaw).transpose(0, 2, 1), R)
    spread = float(np.abs(out - out[0]).max())
    if spread > tol:
        raise ValueError(f"camera axis change is not constant (spread {spread:.2e})")
    return out[0], spread


def pose_from_position_yaw(pos: np.ndarray, yaw: np.ndarray, R_ec: np.ndarray) -> np.ndarray:
    """(N,4,4) camera-to-world poses for commanded positions and headings."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    if len(pos) != len(yaw):
        raise ValueError(f"pos/yaw length mismatch: {len(pos)} vs {len(yaw)}")
    poses = np.tile(np.eye(4), (len(pos), 1, 1))
    poses[:, :3, :3] = yaw_matrix(yaw) @ np.asarray(R_ec, dtype=np.float64)[None]
    poses[:, :3, 3] = pos
    return poses
