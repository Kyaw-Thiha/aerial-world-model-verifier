"""Pinhole camera remapping for generated frames (Track A2 edit-recovery control).

The TII edits use the equidistant fisheye model because that is what the recorded camera
is. ANWM's frames are rendered through a **pinhole** camera at a 90 deg field (f = N/2), so
applying the fisheye map to them would mix a lens change into the rotation being tested.
This module is the pinhole counterpart: same ``cv2.remap`` contract (float32 source maps in
image index coordinates, ``-1`` marks an out-of-image sample), same convention
(camera x right, y down, z optical axis), different projection.

    u, v = pixel_grid(w, h)                     # centred pixel coordinates
    M   = rotation about the camera x axis      # pure camera rotation
    mx, my = pinhole_remap(M, w, h, f=112.0)    # pass to cv2.remap
"""
from __future__ import annotations

import numpy as np

from src.negatives.fisheye_warp import pixel_grid


def rays_pinhole(u: np.ndarray, v: np.ndarray, f: float) -> np.ndarray:
    """Unit rays for centred pinhole pixel coordinates (x right, y down, z optical)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    r = np.stack([u / f, v / f, np.ones_like(u)], axis=-1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def pixels_pinhole(x: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred pixel coordinates and depth for rays; depth <= 0 means unprojectable."""
    x = np.asarray(x, dtype=np.float64)
    z = x[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 1e-9, f * x[..., 0] / np.where(np.abs(z) > 1e-12, z, 1.0), np.nan)
        v = np.where(z > 1e-9, f * x[..., 1] / np.where(np.abs(z) > 1e-12, z, 1.0), np.nan)
    return u, v, z


def pinhole_remap(M: np.ndarray, width: int, height: int, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Source maps for ``cv2.remap`` under a pure camera rotation ``M`` (new = M @ old)."""
    u, v = pixel_grid(width, height)
    x_new = rays_pinhole(u, v, f)
    x_old = np.einsum("ij,hwj->hwi", np.asarray(M, dtype=np.float64), x_new)
    xs, ys, z = pixels_pinhole(x_old, f)
    xi = xs + (width - 1) / 2.0
    yi = ys + (height - 1) / 2.0
    bad = (z <= 1e-9) | ~np.isfinite(xi) | ~np.isfinite(yi) \
        | (xi < -0.5) | (xi > width - 0.5) | (yi < -0.5) | (yi > height - 0.5)
    xi = np.where(bad, -1.0, np.clip(xi, 0.0, width - 1.0)).astype(np.float32)
    yi = np.where(bad, -1.0, np.clip(yi, 0.0, height - 1.0)).astype(np.float32)
    return xi, yi


def applied_tilt_change_deg(M: np.ndarray) -> float:
    """Angle between the camera's optical axis before and after the rotation (degrees)."""
    M = np.asarray(M, dtype=np.float64)
    c = float(np.clip(M[2, 2], -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
