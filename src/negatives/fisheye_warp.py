"""Counterfactual pixel edits: a pure camera rotation in the raw fisheye domain.

Why the fisheye and not the undistorted pinhole view: removing a 40-60 deg body tilt
needs rays well outside the 100 deg virtual pinhole view, and the released calibration
is only invertible to theta = 56 deg, so the edits cannot be expressed with it at all.
The equidistant model ``r = f * theta`` has one parameter, is invertible over the whole
frame, and is the approximation the declared 175 deg horizontal field implies. It is not
fitted to any clip; the pre-registration states the consequence (the null-edit family
measures what the model error costs).

A rotation is applied about the camera centre: for a world-frame rotation ``D`` the
camera-frame map is ``M = R_cw.T @ D @ R_cw``, and ``x_source = M @ x_output``.
"""
from __future__ import annotations

import numpy as np


def equidistant_f(width: int, height: int, fov_deg: float = 175.0) -> float:
    """Focal length (pixels per radian) of an equidistant fisheye at a stated FOV.

    ``fov_deg`` is the horizontal field, so the half-width ``width / 2`` sits at
    ``fov_deg / 2`` from the optical axis.
    """
    return (width / 2.0) / np.deg2rad(fov_deg / 2.0)


def pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates (u, v) as float arrays of shape (height, width)."""
    u = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    v = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    return np.meshgrid(u, v)


def rays_from_pixels(u: np.ndarray, v: np.ndarray, f: float) -> np.ndarray:
    """Unit rays (..., 3) in camera coordinates (x right, y down, z optical axis)."""
    du = np.asarray(u, dtype=np.float64) / f
    dv = np.asarray(v, dtype=np.float64) / f
    theta = np.hypot(du, dv)
    s = np.where(theta > 0, np.sin(theta) / np.where(theta > 0, theta, 1.0), 1.0)
    return np.stack([s * du, s * dv, np.cos(theta)], axis=-1)


def pixels_from_rays(x: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coords and theta for unit rays (..., 3); theta in radians."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.arctan2(np.hypot(x[..., 0], x[..., 1]), x[..., 2])
    r = np.where(theta > 0, theta / np.where(theta > 0, theta, 1.0), 1.0)
    # direction of the ray's projection onto the image plane, unit length
    rho = np.hypot(x[..., 0], x[..., 1])
    cos_a = np.where(rho > 0, x[..., 0] / np.where(rho > 0, rho, 1.0), 1.0)
    sin_a = np.where(rho > 0, x[..., 1] / np.where(rho > 0, rho, 1.0), 0.0)
    return f * theta * cos_a, f * theta * sin_a, theta


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample of ``img`` (h, w) or (h, w, c) at floating-point pixel coords.

    Returns ``(values, weight_sum)``; ``weight_sum`` is 1 where the sample was fully
    inside the image, so callers can build a validity mask from it.
    """
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    x1, y1 = x0 + 1, y0 + 1
    ok = {
        (0, 0): (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h),
        (0, 1): (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h),
        (1, 0): (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h),
        (1, 1): (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h),
    }
    out = np.zeros(xs.shape + img.shape[2:], dtype=np.float64)
    acc = np.zeros(xs.shape, dtype=np.float64)
    for (dx, dy) in ok:
        valid = ok[(dx, dy)]
        wx = fx if dx else 1.0 - fx
        wy = fy if dy else 1.0 - fy
        wgt = wx * wy
        rows = np.where(valid, y0 + dy, 0)
        cols = np.where(valid, x0 + dx, 0)
        sampled = img[rows, cols].astype(np.float64)
        add = sampled * (wgt[..., None] if sampled.ndim > 2 else wgt)
        out += np.where(valid, add, 0.0)
        acc += np.where(valid, wgt, 0.0)
    return out, acc


def warp_fisheye(img: np.ndarray, M: np.ndarray, f: float | None = None,
                 fov_deg: float = 175.0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a fisheye frame by the camera-frame rotation ``M`` about its centre.

    Returns ``(warped_uint8, valid_mask)``. Pixels whose source falls outside the image
    are filled with 0 (black) and marked False; ``valid_mask.mean()`` is the fraction of
    the output that carries real content.
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    if f is None:
        f = equidistant_f(w, h, fov_deg)
    u, v = pixel_grid(w, h)
    x_new = rays_from_pixels(u, v, f)
    # x_source = M @ x_output: M is camera-frame, M = R_cw^T D R_cw for world rotation D
    x_old = np.einsum("ij,hwj->hwi", np.asarray(M, dtype=np.float64), x_new)
    xs, ys, _ = pixels_from_rays(x_old, f)
    # pixel coordinates -> image indices (centre of the image is at (w-1)/2)
    xi = xs + (w - 1) / 2.0
    yi = ys + (h - 1) / 2.0
    sub = img.astype(np.float64)
    out, acc = _bilinear(sub, xi, yi)
    valid = acc > 1e-9
    if out.ndim > 2:
        out = np.where(valid[..., None], out, 0.0)
    else:
        out = np.where(valid, out, 0.0)
    if np.issubdtype(img.dtype, np.integer):
        out = np.clip(np.rint(out), 0, 255).astype(img.dtype)
    return out, valid


def rotate_pixels(img: np.ndarray, M: np.ndarray, f: float | None = None,
                  fov_deg: float = 175.0) -> np.ndarray:
    """``warp_fisheye`` returning only the image (for remote use with cv2.remap)."""
    return warp_fisheye(img, M, f, fov_deg)[0]


def remap_arrays(M: np.ndarray, width: int, height: int, f: float | None = None,
                 fov_deg: float = 175.0) -> tuple[np.ndarray, np.ndarray]:
    """Source-coordinate maps for ``cv2.remap`` (fast path used on the GPU worker).

    Same geometry as :func:`warp_fisheye`; returns float32 ``map_x, map_y`` in image
    index coordinates (callers pass ``borderValue=0`` and build the validity mask from
    ``-1`` entries by masking out-of-range coordinates).
    """
    if f is None:
        f = equidistant_f(width, height, fov_deg)
    u, v = pixel_grid(width, height)
    x_new = rays_from_pixels(u, v, f)
    x_old = np.einsum("ij,hwj->hwi", np.asarray(M, dtype=np.float64), x_new)
    xs, ys, _ = pixels_from_rays(x_old, f)
    xi = (xs + (width - 1) / 2.0).astype(np.float32)
    yi = (ys + (height - 1) / 2.0).astype(np.float32)
    # half-pixel tolerance: bilinear sampling is defined over [-0.5, size - 0.5], and
    # the identity map lands on -1e-15 for the first pixel, which must stay valid
    bad = ((xi < -0.5) | (xi > width - 0.5) | (yi < -0.5) | (yi > height - 0.5))
    # invalid samples get -1 (cv2.remap then writes the border value); valid ones are
    # clipped into the image so a tiny negative round-off never looks invalid
    xi = np.where(bad, -1.0, np.clip(xi, 0.0, width - 1.0)).astype(np.float32)
    yi = np.where(bad, -1.0, np.clip(yi, 0.0, height - 1.0)).astype(np.float32)
    return xi, yi


def warp_with_remap(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.remap`` wrapper; returns (image, valid mask) with black borders."""
    import cv2
    import numpy as np

    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (map_x >= 0) & (map_y >= 0)
    return out, valid
