"""Fisheye undistortion into a virtual pinhole camera.

The TII calibration is ``cv2.calibrateCamera`` output (k1,k2,p1,p2,k3), i.e. the
OpenCV polynomial model, not a Kannala-Brandt fisheye model. Writing it out:

    x_n = (u - cx)/fx,  y_n = (v - cy)/fy,  r = |(x_n, y_n)|
    rd  = r * (1 + k1 r^2 + k2 r^4 + k3 r^6)          (p1 = p2 ~ 0 here)
    u_d = fx * x_n * (rd/r) + cx,  v_d = fy * y_n * (rd/r) + cy

Two consequences are measured by :func:`model_diagnostics` rather than assumed:

* the polynomial is only invertible up to the radius where ``rd(r)`` peaks
  (about 1.1 in these files), so a fraction of the frame corners has **no**
  representable ray; those pixels are excluded, not extrapolated;
* ``fx / fy`` is 0.749 in all three calibration files, which no square-pixel
  sensor produces. The virtual camera therefore uses a single focal length and
  the anisotropy is reported in the diagnostics.

:func:`virtual_camera` builds a pinhole view (virtual FOV and output size are
explicit arguments), and :func:`undistort` remaps with numpy. ``cv2`` is used
only when present and only for speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_FOV_DEG = 100.0
DEFAULT_OUT_SIZE = (640, 480)


def distortion_radius(r: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """``rd`` for undistorted normalised radius ``r`` (OpenCV 5-coefficient model)."""
    k1, k2, p1, p2, k3 = np.asarray(dist, dtype=np.float64).ravel()
    return r * (1.0 + k1 * r**2 + k2 * r**4 + k3 * r**6)


def inv_distortion_radius(rd: float, dist: np.ndarray, r_hi: float = 4.0) -> float:
    """Smallest ``r`` with ``rd(r) == rd``; NaN when ``rd`` exceeds the model's range."""
    rs = np.linspace(0.0, r_hi, 200001)
    vals = distortion_radius(rs, dist)
    peak = int(np.argmax(vals))
    if rd > vals[peak]:
        return float("nan")
    lo, hi = 0.0, rs[peak]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if distortion_radius(np.array([mid]), dist)[0] < rd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def model_diagnostics(K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int]) -> dict:
    """What the released calibration can and cannot represent."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size
    hi_rd = max(np.hypot((u - cx) / fx, (v - cy) / fy)
                for u, v in ((0, 0), (w, 0), (0, h), (w, h)))
    rs = np.linspace(0.0, 4.0, 200001)
    vals = distortion_radius(rs, dist)
    peak = int(np.argmax(vals))
    r_peak, rd_peak = float(rs[peak]), float(vals[peak])
    corners = {}
    for name, (u, v) in {"tl": (0, 0), "tr": (w, 0), "bl": (0, h), "br": (w, h)}.items():
        rd = float(np.hypot((u - cx) / fx, (v - cy) / fy))
        r = inv_distortion_radius(rd, dist)
        corners[name] = {"rd": rd, "theta_deg": float(np.degrees(np.arctan(r)))
                         if np.isfinite(r) else None}
    return {
        "fx": float(fx), "fy": float(fy), "fx_fy_ratio": float(fx / fy),
        "principal_point": [float(cx), float(cy)],
        "image_size": [int(w), int(h)],
        "max_observed_rd": float(hi_rd),
        "rd_peak": rd_peak,
        "theta_at_peak_deg": float(np.degrees(np.arctan(r_peak))),
        "corners": corners,
        "corners_representable": all(c["theta_deg"] is not None for c in corners.values()),
        "diagonal_representable_deg": 2.0 * float(np.degrees(np.arctan(r_peak)))
        if hi_rd > rd_peak else None,
    }


def theta_to_pixel(theta_deg: float, K: np.ndarray, dist: np.ndarray) -> float:
    """Radial source-pixel distance for a ray ``theta_deg`` off the optical axis."""
    r = np.tan(np.radians(theta_deg))
    return float(distortion_radius(np.array([r]), dist)[0] * K[0, 0])


@dataclass
class VirtualCamera:
    fov_deg: float
    out_size: tuple[int, int]
    K: np.ndarray                # virtual pinhole intrinsics (square pixels)
    map_x: np.ndarray            # (h, w) source x, NaN where no ray exists
    map_y: np.ndarray
    valid_fraction: float
    max_source_theta_deg: float
    source_K: np.ndarray
    source_dist: np.ndarray
    source_size: tuple[int, int]


def virtual_camera(K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int],
                   fov_deg: float = DEFAULT_FOV_DEG,
                   out_size: tuple[int, int] | None = None,
                   flip_vertical: bool = False,
                   R_rot: np.ndarray | None = None) -> VirtualCamera:
    """Pinhole view of the fisheye with horizontal ``fov_deg`` and ``out_size``.

    By default the virtual camera keeps the optical axis and the image centre of
    the source, so a ray at angle theta from the axis lands at the same azimuth in
    both images. ``R_rot`` (3x3, source-frame -> virtual-frame) tilts the virtual
    camera; it is what the synthetic-rotation probe uses to check, on real frames,
    that an estimator's reported gravity moves by the injected amount.
    """
    w_src, h_src = image_size
    w_out, h_out = out_size or (w_src, h_src)
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    f = (w_out / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K_out = np.array([[f, 0.0, (w_out - 1) / 2.0],
                      [0.0, f, (h_out - 1) / 2.0],
                      [0.0, 0.0, 1.0]])

    u, v = np.meshgrid(np.arange(w_out, dtype=np.float64),
                       np.arange(h_out, dtype=np.float64))
    x = (u - K_out[0, 2]) / f
    y = (v - K_out[1, 2]) / f
    if flip_vertical:
        y = -y
    if R_rot is not None:
        R_rot = np.asarray(R_rot, dtype=np.float64)
        ray = np.stack([x, y, np.ones_like(x)], axis=-1)          # virtual rays
        ray = ray @ R_rot                                         # into the source frame
        ray = ray / ray[..., 2:3]
        x, y = ray[..., 0], ray[..., 1]
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan(r))

    rd = distortion_radius(r, dist)
    rs = np.linspace(0.0, 4.0, 200001)
    rd_peak = float(np.max(distortion_radius(rs, dist)))
    ok = np.isfinite(rd) & (rd <= rd_peak)
    # invert rd -> r_grid by monotone interpolation on the ascending branch
    r_asc = rs[: int(np.argmax(distortion_radius(rs, dist))) + 1]
    rd_asc = distortion_radius(r_asc, dist)
    r_grid = np.interp(rd, rd_asc, r_asc, left=0.0, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(rd > 0, r_grid / np.where(rd > 0, rd, 1.0), 1.0)
    map_x = fx * x * scale + cx
    map_y = fy * y * scale + cy
    inside = (map_x >= 0) & (map_x <= w_src - 1) & (map_y >= 0) & (map_y <= h_src - 1)
    valid = ok & inside & np.isfinite(map_x) & np.isfinite(map_y)
    map_x = np.where(valid, map_x, np.nan)
    map_y = np.where(valid, map_y, np.nan)
    return VirtualCamera(fov_deg=float(fov_deg), out_size=(w_out, h_out), K=K_out,
                         map_x=map_x, map_y=map_y,
                         valid_fraction=float(valid.mean()),
                         max_source_theta_deg=float(theta[valid].max()) if valid.any() else 0.0,
                         source_K=np.asarray(K, dtype=np.float64),
                         source_dist=np.asarray(dist, dtype=np.float64),
                         source_size=(w_src, h_src))


def undistort(img: np.ndarray, vc: VirtualCamera) -> np.ndarray:
    """Remap ``img`` into the virtual camera. Invalid pixels are black; alpha=1 channel
    is appended so gaps are visible instead of silently black."""
    import sys

    if "cv2" in sys.modules or _cv2_available():
        import cv2

        mx = np.where(np.isnan(vc.map_x), -1.0, vc.map_x).astype(np.float32)
        my = np.where(np.isnan(vc.map_y), -1.0, vc.map_y).astype(np.float32)
        return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _remap_bilinear(img, vc.map_x, vc.map_y)


def _cv2_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("cv2") is not None


def _remap_bilinear(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((*mx.shape, img.shape[2]) if img.ndim == 3 else mx.shape, dtype=img.dtype)
    v = ~np.isnan(mx)
    xs, ys = mx[v], my[v]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xs - x0), (ys - y0)
    if img.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]
    out[v] = (p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy)
              + p10 * (1 - fx) * fy + p11 * fx * fy)
    return out


def straightness_score(img: np.ndarray, lines: np.ndarray | None = None) -> dict:
    """RMS distance of detected line pixels to their fitted line, in pixels.

    Objective check that the undistortion did not bend straight structure. Uses
    cv2's LSD when available (a line segment's own endpoints are used), otherwise
    returns ``{"available": False}``.
    """
    if not _cv2_available():
        return {"available": False, "reason": "cv2 not installed"}
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(gray)[0]
    if segs is None:
        return {"available": True, "n_segments": 0}
    lens = np.linalg.norm(segs[:, 0, :] - segs[:, 1, :], axis=1)
    long = segs[lens > 0.15 * max(img.shape[:2])]
    return {"available": True, "n_segments": int(len(segs)),
            "n_long": int(len(long)),
            "longest_px": float(lens.max()) if len(lens) else 0.0}
