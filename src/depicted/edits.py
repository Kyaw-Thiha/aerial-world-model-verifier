"""Pure-rotation pixel edits for UZH-FPV, with a common border mask.

A rotation about the camera centre creates no parallax, so an edit is an exact remapping of
rays. Two geometries are supported because the two primary estimator variants read different
frames:

* ``pinhole`` — the undistorted view used by variants a and c. A rotation is the homography
  ``x_source = K M K^-1 x_output``.
* ``kb4``     — the raw fisheye used by variant b, with the rig's own Kalibr
  pinhole+equidistant (KB4) model, inverted by monotone lookup. This is *the dataset's own*
  model, unlike Track A's single-parameter equidistant approximation, so the edit is exact on
  the released calibration.

``M`` is the camera-frame rotation with the convention of ``src.negatives.fisheye_warp``:
``x_source = M @ x_output``, i.e. ``M = R_cw^T D R_cw`` for a world rotation ``D``. The
builders in ``src.negatives.edits`` produce exactly that, and are imported read-only.

Border handling: every condition of a frame is masked with the **intersection** of the
conditions' valid regions, and the identity ("null edit") condition is masked the same way.
Without that, the null edit would keep pixels the tilt edits lose and the estimator would see
a border difference it can turn into a tilt difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- KB4 model
class KB4Camera:
    """pinhole + equidistant (Kannala-Brandt) camera, OpenCV ``fisheye`` convention."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, k: list[float]):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k = np.asarray(k, dtype=np.float64).ravel()
        if self.k.size < 4:
            self.k = np.pad(self.k, (0, 4 - self.k.size))
        # The KB4 polynomial is only invertible where theta_d is increasing. For several of
        # these cameras it turns over inside the frame (indoor_45, outdoor_forward), and an
        # inversion that ignores the turnover collapses the image onto a radial smear — which
        # is what a null edit produced before this was fixed. The lookup is therefore built on
        # the increasing branch only, and radii beyond it are declared outside the model.
        theta = np.linspace(0.0, np.pi, 20001)
        theta_d = self._distort(theta)
        keep = np.ones(len(theta), bool)
        dturn = np.diff(theta_d)
        bad = np.flatnonzero(dturn <= 0)
        if len(bad):
            keep[bad[0] + 1:] = False
        self._theta = theta[keep]
        self._theta_d = theta_d[keep]
        self.max_theta_rad = float(self._theta[-1])
        self.max_theta_d = float(self._theta_d[-1])

    def _distort(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        poly = 1.0 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3])))
        return theta * poly

    def pixels_from_rays(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(…, 3) camera-frame rays -> (u, v) pixel coordinates."""
        x = np.asarray(x, dtype=np.float64)
        theta = np.arctan2(np.hypot(x[..., 0], x[..., 1]), x[..., 2])
        rho = np.hypot(x[..., 0], x[..., 1])
        ca = np.where(rho > 0, x[..., 0] / np.where(rho > 0, rho, 1.0), 1.0)
        sa = np.where(rho > 0, x[..., 1] / np.where(rho > 0, rho, 1.0), 0.0)
        rd = self._distort(theta)
        return self.fx * rd * ca + self.cx, self.fy * rd * sa + self.cy

    def pixels_from_rays_checked(self, x: np.ndarray):
        """Pixels **and** an inside-the-model mask for (…, 3) rays."""
        x = np.asarray(x, dtype=np.float64)
        theta = np.arctan2(np.hypot(x[..., 0], x[..., 1]), x[..., 2])
        u, v = self.pixels_from_rays(x)
        return u, v, theta <= self.max_theta_rad

    def rays_from_pixels(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(u, v) pixel coordinates -> unit rays (…, 3); clamps at the model's limit."""
        a = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        b = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        rd = np.hypot(a, b)
        theta = np.interp(rd, self._theta_d, self._theta)
        r = np.tan(theta)
        scale = np.where(rd > 0, r / np.where(rd > 0, rd, 1.0), 1.0)
        x = a * scale
        y = b * scale
        z = np.ones_like(x)
        n = np.sqrt(x * x + y * y + z * z)
        return np.stack([x / n, y / n, z / n], axis=-1)


def kb4_from_meta(meta: dict) -> KB4Camera:
    c = meta["intrinsics_kb4"]
    return KB4Camera(c["fx"], c["fy"], c["cx"], c["cy"], c["k"])


def pinhole_K(meta: dict) -> np.ndarray:
    c = meta["intrinsics_pinhole"]
    return np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- sampling maps
def _pixel_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return u, v


def pinhole_source_coords(M: np.ndarray, K: np.ndarray, w: int, h: int):
    """Source pixel coordinates for every output pixel under the rotation ``M``."""
    u, v = _pixel_grid(w, h)
    x_new = np.stack([u, v, np.ones_like(u)], axis=-1)
    K_inv = np.linalg.inv(K)
    x_old = np.einsum("ij,hwj->hwi", K @ np.asarray(M, dtype=np.float64) @ K_inv, x_new)
    z = x_old[..., 2]
    ok = z > 1e-9
    zs = np.where(ok, z, 1.0)
    return np.where(ok, x_old[..., 0] / zs, -1.0), np.where(ok, x_old[..., 1] / zs, -1.0), ok


def kb4_source_coords(M: np.ndarray, cam: KB4Camera, w: int, h: int):
    u, v = _pixel_grid(w, h)
    x_new = cam.rays_from_pixels(u, v)
    x_old = np.einsum("ij,hwj->hwi", np.asarray(M, dtype=np.float64), x_new)
    su, sv, inside_model = cam.pixels_from_rays_checked(x_old)
    ok = (x_old[..., 2] > 1e-6) & inside_model
    return su, sv, ok


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    from src.negatives.fisheye_warp import _bilinear as bil

    return bil(img, xs, ys)


def warp(img: np.ndarray, M: np.ndarray, geometry: str, calib, mask: np.ndarray | None = None):
    """Rotate one frame; returns ``(warped_uint8_or_float, valid)``.

    ``geometry`` is ``"pinhole"`` (``calib`` = 3x3 K) or ``"kb4"`` (``calib`` = KB4Camera).
    ``mask`` optionally ANDs one more validity map (the per-frame common mask).
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    if geometry == "pinhole":
        xs, ys, ok = pinhole_source_coords(M, calib, w, h)
    elif geometry == "kb4":
        xs, ys, ok = kb4_source_coords(M, calib, w, h)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    out, acc = _bilinear(img.astype(np.float64), xs, ys)
    valid = (acc > 1e-9) & ok
    if mask is not None:
        valid &= mask
    if out.ndim > 2:
        out = np.where(valid[..., None], out, 0.0)
    else:
        out = np.where(valid, out, 0.0)
    if np.issubdtype(img.dtype, np.integer):
        out = np.clip(np.rint(out), 0, 255).astype(img.dtype)
    return out, valid


def common_mask(maps: list[np.ndarray], w: int, h: int, geometry: str, calib,
                min_valid: float = 0.0) -> np.ndarray:
    """Validity mask shared by every condition of one frame (identity included).

    The intersection is taken over the conditions actually exported, so the null edit loses
    exactly the pixels the strongest edit loses.
    """
    masks = []
    for M in maps:
        if geometry == "pinhole":
            xs, ys, ok = pinhole_source_coords(M, calib, w, h)
        else:
            xs, ys, ok = kb4_source_coords(M, calib, w, h)
        inside = ok & (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        masks.append(inside)
    m = np.logical_and.reduce(masks)
    if m.mean() < min_valid:
        raise ValueError(f"common mask keeps only {m.mean():.2f} of the frame")
    return m


# --------------------------------------------------------------------------- conditions
@dataclass
class Condition:
    name: str
    k: float = 1.0            # tilt scale, used when name == "tilt_k"
    yaw_deg: float = 0.0      # used when name == "yaw_control"


def camera_maps(cond: Condition, R_bw: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """Camera-frame rotations ``M`` for every frame of a clip under one condition."""
    from src.negatives.edits import condition_maps

    R_cw = np.einsum("nij,jk->nik", np.asarray(R_bw, dtype=np.float64),
                     np.asarray(R_cb, dtype=np.float64))
    if cond.name == "identity":
        return condition_maps("identity", R_bw, R_cw)
    if cond.name == "null_edit":
        return condition_maps("identity", R_bw, R_cw)
    if cond.name == "mirror_roll":
        return condition_maps("mirror_roll", R_bw, R_cw)
    if cond.name == "yaw_control":
        return condition_maps("yaw_control", R_bw, R_cw, yaw_rot_rad=np.radians(cond.yaw_deg))
    if cond.name == "tilt_k":
        return condition_maps("tilt_k", R_bw, R_cw, k=cond.k)
    if cond.name.startswith("delay_ms:"):
        return condition_maps("identity", R_bw, R_cw)
    raise ValueError(f"unknown condition {cond.name!r}")


def default_conditions(yaw_deg: float = 20.0) -> list[Condition]:
    """The six exported conditions of the shared clip schema."""
    return [
        Condition("real"),
        Condition("null_edit"),
        Condition("tilt_removal", k=0.0),
        Condition("scale_0.5", k=0.5),
        Condition("scale_1.5", k=1.5),
        Condition("mirror_roll"),
        Condition("yaw_control", yaw_deg=yaw_deg),
    ]
