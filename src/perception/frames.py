"""Camera frames: extrinsics, gravity direction in the camera, projected overlays.

Conventions, stated once and used everywhere (all are checked by tests):

* world  : z up, metres (``docs/project_structure.md``)
* body   : FLU, x forward, y left, z up (TII FCU IMU frame, verified in
           ``src/datasets/tii.py``)
* camera : OpenCV RDF, x right, y down, z forward (optical axis)
* ``R_cb``: rotation whose columns are the camera axes expressed in body
  coordinates, i.e. it maps camera->body. Its transpose maps body->camera.
* "up" (gravity reference) is the unit vector pointing away from gravity. In
  camera coordinates it is ``R_cb^T @ R_bw^T @ e3`` with ``R_bw`` the mocap
  body-to-world rotation.
* body tilt = angle between the body z axis and world up
  = ``arccos(up_body . e3)``; roll and pitch follow from ``R`` in FLU.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

E3 = np.array([0.0, 0.0, 1.0])


def body_to_world(quat: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    """(N, 4) quaternions -> (N, 3, 3) body-to-world rotation matrices."""
    q = np.asarray(quat, dtype=np.float64)
    if quat_order == "wxyz":
        q = q[:, [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"unknown quat_order {quat_order!r}")
    return Rotation.from_quat(q).as_matrix()


def up_in_body(R_bw: np.ndarray) -> np.ndarray:
    """World up expressed in body coordinates, (N, 3)."""
    return np.einsum("nji,j->ni", R_bw, E3)


def up_in_camera(R_bw: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """World up expressed in camera coordinates, (N, 3).

    ``R_cb`` is either one (3, 3) matrix for the whole clip or (N, 3, 3).
    """
    R_cb = np.asarray(R_cb, dtype=np.float64)
    up_b = up_in_body(R_bw)
    if R_cb.ndim == 2:
        return up_b @ R_cb            # == R_cb^T @ up_b per row
    return np.einsum("nji,nj->ni", R_cb, up_b)


def body_z_in_camera(R_cb: np.ndarray) -> np.ndarray:
    """Body z axis (thrust/up axis of the airframe) expressed in camera coords.

    ``R_cb`` maps camera->body, so a body vector maps into the camera frame through
    its transpose; the third *row* of ``R_cb`` is therefore body z in camera coords.
    (The third *column* is the camera optical axis in body coords — see
    :func:`optical_axis_in_body`.)
    """
    R = np.asarray(R_cb, dtype=np.float64)
    return R[..., 2, :] if R.ndim == 3 else R[2, :]


def optical_axis_in_body(R_cb: np.ndarray) -> np.ndarray:
    """Camera optical axis (+z) expressed in body coordinates."""
    R = np.asarray(R_cb, dtype=np.float64)
    return R[..., :, 2] if R.ndim == 3 else R[:, 2]


def tilt_deg(up_body: np.ndarray) -> np.ndarray:
    """Body tilt = angle between body z and world up, degrees."""
    return np.degrees(np.arccos(np.clip(up_body[:, 2], -1.0, 1.0)))


def roll_pitch_deg(R_bw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-Y-X (yaw-pitch-roll) decomposition of body-to-world, degrees."""
    ypr = Rotation.from_matrix(R_bw).as_euler("ZYX", degrees=True)
    return ypr[:, 2], ypr[:, 1]


def project_direction(u: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project directions (N, 3) in camera coords to pixels; mask where u_z <= 0."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    K = np.asarray(K, dtype=np.float64)
    z = u[:, 2]
    ok = z > 1e-9
    safe = np.where(ok, z, 1.0)
    x = K[0, 0] * u[:, 0] / safe + K[0, 2]
    y = K[1, 1] * u[:, 1] / safe + K[1, 2]
    return np.column_stack([x, y]), ok


def horizon_points(up_cam: np.ndarray, K: np.ndarray, n: int = 720) -> np.ndarray:
    """Points of the horizon (great circle perpendicular to ``up_cam``) in pixels.

    Returns an (M, 2) array of points in front of the camera, so it can be drawn
    as a polyline together with the image border.
    """
    up = np.asarray(up_cam, dtype=np.float64)
    up = up / np.linalg.norm(up)
    a = np.cross(up, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(up, np.array([1.0, 0.0, 0.0]))
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dirs = np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b
    px, ok = project_direction(dirs, K)
    return px[ok]


def expected_gravity_offset_deg(R_cb: np.ndarray) -> float:
    """Angle between world up and the camera optical axis for a level airframe.

    Also the angle between the body z axis and the optical axis (the mount angle's
    complement), computed from ``R_cb[2, 2]`` so it does not depend on which of the
    two functions above is called by mistake.
    """
    R = np.asarray(R_cb, dtype=np.float64)
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1.0, 1.0))))


def camera_to_body_pitched(theta_rad: float) -> np.ndarray:
    """Camera->body rotation for a hard-mounted camera pitched up by ``theta_rad``.

    Column layout is the camera axes in body coordinates: x (right) = -body y,
    y (down) = (sin, 0, -cos), z (optical axis) = (cos, 0, sin). For theta = 50 deg
    this reproduces the released ``ellipse``/``trackRATM`` matrix exactly, which
    is the evidence that ``drone_to_camera.json`` is camera->body.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[0.0, s, c],
                     [-1.0, 0.0, 0.0],
                     [0.0, -c, s]])
