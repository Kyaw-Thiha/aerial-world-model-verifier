"""Vertical-motion defects on the frozen simulated paths (Track V).

The frozen set (Track I) labels camera *attitude* defects: the path is fixed and the camera's
attitude is edited. This module adds the orthogonal family: the path (and therefore the
commanded altitude) is fixed and the camera's *altitude* is edited, with the camera attitude
left at the flatness attitude of the commanded path. That isolates the vertical axis: the tilt
check sees a feasible clip, so only a vertical-motion check can react.

Edits (``kind``), applied to ``flight.pos[:, 2]`` over the whole clip, with ``z0`` the commanded
start altitude (12 m in the frozen set):

* ``removed``    -- z(t) = z0                       (level flight along the same ground track)
* ``scaled_0.5`` -- z(t) = z0 + 0.5 (z - z0)        (climbs half as much)
* ``scaled_1.5`` -- z(t) = z0 + 1.5 (z - z0)        (climbs 1.5x as much)
* ``reversed``   -- z(t) = z0 - m (z - z0), m as large as the ground clearance allows
                    (m = min(1, (z0 - clearance) / (z_max - z0))); the realised m is recorded
* ``yaw``        -- camera rotated about world up by ``yaw_deg``; z unchanged (control)
* ``null``       -- nothing changed (control; the re-render must reproduce the frozen clip)

The commanded path is never edited: the depth-ratio command the check compares against comes
from the frozen truth, and the defect is exactly the difference between the two.
"""
from __future__ import annotations

import numpy as np

EDIT_KINDS = ("null", "yaw", "removed", "scaled_0.5", "scaled_1.5", "reversed")
GROUND_CLEARANCE_M = 1.5
YAW_CONTROL_DEG = 25.0


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply_edit(clip, kind: str, yaw_deg: float = YAW_CONTROL_DEG,
               clearance: float = GROUND_CLEARANCE_M) -> dict:
    """Edit ``clip.flight.pos`` in place; return the metadata that documents the edit."""
    if kind not in EDIT_KINDS:
        raise ValueError(f"unknown edit {kind}")
    z = np.asarray(clip.flight.pos[:, 2], dtype=np.float64).copy()
    z0 = float(z[0])
    meta = {"kind": kind, "z0_m": z0, "z_commanded_min": float(z.min()),
            "z_commanded_max": float(z.max()), "clearance_m": clearance,
            "yaw_deg": 0.0, "reversed_factor": None}
    if kind == "removed":
        z_new = np.full_like(z, z0)
    elif kind == "scaled_0.5":
        z_new = z0 + 0.5 * (z - z0)
    elif kind == "scaled_1.5":
        z_new = z0 + 1.5 * (z - z0)
    elif kind == "reversed":
        m = min(1.0, (z0 - clearance) / max(float(z.max() - z0), 1e-9))
        z_new = z0 - m * (z - z0)
        meta["reversed_factor"] = float(m)
    else:                                   # null, yaw
        z_new = z.copy()
    if kind == "yaw":
        meta["yaw_deg"] = float(yaw_deg)
    clip.flight.pos[:, 2] = z_new
    meta["z_rendered_min"] = float(z_new.min())
    meta["z_rendered_max"] = float(z_new.max())
    meta["rendered_ratio"] = float((z_new[-1] - z0) / (z[-1] - z0)) if abs(z[-1] - z0) > 1e-9 \
        else None
    return meta


def edited_poses(clip, kind: str, yaw_deg: float = YAW_CONTROL_DEG) -> np.ndarray:
    """Camera-to-world poses for the edited clip (yaw control rotates about world up)."""
    poses = clip.poses()
    if kind == "yaw":
        R = _rz(yaw_deg)
        out = poses.copy()
        out[:, :3, :3] = R[None, :, :] @ poses[:, :3, :3]
        out[:, :3, 3] = poses[:, :3, 3] @ R.T
        return out
    return poses
