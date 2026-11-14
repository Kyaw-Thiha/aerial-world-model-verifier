"""Dense-rate required lean and truth-attitude lean (schema v2 support).

Why this exists.  The stored per-frame grid is 12 Hz (stride 10), where a 0.37 s
Savitzky-Golay window is 5 samples.  The commanded path itself is known at the mocap rate
(274.73 Hz), so differentiating there and sampling at the frame times removes most of the
second-derivative noise that attenuates the gain regression (the real-clip gain was 0.745
instead of about 1).  Schema v2 carries those arrays as ``lean_req`` and ``vel_req``;
``pos`` and ``yaw`` stay for provenance and for callers that cannot supply them.

All arrays are in the yaw-aligned frame:
``l = (cos(phi) sin(theta), -sin(phi))`` for the unit thrust axis, and
``vel_req = Rz(-yaw) v_horizontal / g`` (units 1/s when multiplied by a drag coefficient).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.physics.flatness import G0

from .lean import world_to_yaw_frame

E3 = np.array([0.0, 0.0, 1.0])
SG_WINDOW_S = 0.37
SG_POLY = 3


@dataclass
class DenseRequired:
    l_req0: np.ndarray        # (N, 2) drag-free required lean at the requested times
    vel_req: np.ndarray       # (N, 2) horizontal velocity / g at the requested times
    tilt_deg: np.ndarray      # (N,) required tilt from the unit thrust axis
    dt_s: float               # native sample interval
    window: int               # SG window in native samples
    masked_frames: int        # requested frames outside the native interior (now NaN)


def _odd_window(n: int, dt_s: float, window_s: float, polyorder: int) -> int:
    w = int(round(window_s / dt_s))
    if w % 2 == 0:
        w += 1
    w = max(polyorder + 2, w)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w


def dense_required_lean(mocap_path: str | Path, t_frame: np.ndarray,
                        window_s: float = SG_WINDOW_S, polyorder: int = SG_POLY,
                        g: float = G0) -> DenseRequired:
    """Differentiate the native-rate mocap path and sample the required lean at ``t_frame``.

    ``t_frame`` must be on the same relative clock as the mocap file, which is how the
    TII video loader and ``phase3a_make_specs`` already use it.  Frames outside the
    native SG interior (half a window at each end) are NaN, never extrapolated.
    """
    clip = load_tii(mocap_path, name=Path(mocap_path).stem)
    t_native = np.asarray(clip.t, dtype=np.float64)
    dt_s = 1.0 / float(clip.fps)
    n = len(t_native)
    w = _odd_window(n, dt_s, window_s, polyorder)
    vel = savgol_filter(clip.pos, w, polyorder, deriv=1, delta=dt_s, axis=0,
                        mode="interp")
    acc = savgol_filter(clip.pos, w, polyorder, deriv=2, delta=dt_s, axis=0,
                        mode="interp")
    q = acc + g * E3
    norm = np.linalg.norm(q, axis=1)
    u0 = np.full_like(q, np.nan)
    good = np.isfinite(norm) & (norm > 1e-9)
    u0[good] = q[good] / norm[good, None]
    R_bw = Rotation.from_quat(clip.quat).as_matrix()
    yaw_native = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
    l_native = world_to_yaw_frame(u0[:, :2], yaw_native)
    v_native = world_to_yaw_frame(vel[:, :2] / g, yaw_native)
    tilt_native = np.degrees(np.arccos(np.clip(u0[:, 2], -1.0, 1.0)))
    half = w // 2
    l_native[:half] = np.nan
    l_native[n - half:] = np.nan
    v_native[:half] = np.nan
    v_native[n - half:] = np.nan
    tilt_native[:half] = np.nan
    tilt_native[n - half:] = np.nan

    t_frame = np.asarray(t_frame, dtype=np.float64)
    tt = np.clip(t_frame, t_native[0], t_native[-1])

    def sample(arr: np.ndarray) -> np.ndarray:
        out = np.column_stack([np.interp(tt, t_native, arr[:, k])
                               for k in range(arr.shape[1])])
        # np.interp does not preserve NaN; restore frames whose bracketing native
        # samples are non-finite or outside the interior.
        idx = np.searchsorted(t_native, tt)
        idx = np.clip(idx, 1, n - 1)
        bad = (~np.isfinite(arr[idx - 1]).all(axis=1)
               | ~np.isfinite(arr[idx]).all(axis=1))
        out[bad] = np.nan
        return out

    l_req0 = sample(l_native)
    vel_req = sample(v_native)
    tilt = np.interp(tt, t_native, tilt_native)
    idx = np.clip(np.searchsorted(t_native, tt), 1, n - 1)
    bad = ~np.isfinite(tilt_native[idx - 1]) | ~np.isfinite(tilt_native[idx])
    tilt[bad] = np.nan
    return DenseRequired(l_req0=l_req0, vel_req=vel_req, tilt_deg=tilt,
                         dt_s=dt_s, window=w,
                         masked_frames=int(np.sum(~np.isfinite(l_req0).all(axis=1))))


def truth_lean(quat: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truth-attitude lean and tilt from the mocap body-to-world rotation.

    ``quat`` is scipy order (xyzw), as the TII loader stores it.  The body z-axis is the
    thrust axis; its horizontal part in the yaw frame is the lean.  No gravity estimate,
    no mount, no camera: this is the attitude the airframe actually had.
    """
    R_bw = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    z_b = R_bw[:, :, 2]
    lean = world_to_yaw_frame(z_b[:, :2], np.asarray(yaw, dtype=np.float64))
    tilt = np.degrees(np.arccos(np.clip(z_b[:, 2], -1.0, 1.0)))
    return lean, tilt
