"""Native-rate truth kinematics for the two datasets (Track K, Part 2).

One :class:`Kinematics` per TII flight or UZH sequence, on the native truth clock:

* velocity, acceleration and jerk from a Savitzky-Golay fit of the truth position,
* heading and the actual body z axis from the truth attitude,
* the actual yaw-aligned lean vector ``l = Rz(-yaw) (z_x, z_y)``.

The first and last half window are masked (``valid`` False), never extrapolated.  UZH
streams are truncated to the image-frame time span of the sequence, i.e. the footage
the verdict uses, not the whole recording.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.datasets.tii_video import flight_parts
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[2]
TII_FLIGHTS = [
    "flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
    "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
    "flight-13p-trackRATM", "flight-16p-trackRATM",
]
UZH_FORWARD = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
               "indoor_forward_7", "indoor_forward_9", "indoor_forward_10",
               "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5"]
UZH_45 = ["indoor_45_2", "indoor_45_4", "indoor_45_9", "indoor_45_13", "outdoor_45_1"]


@dataclass
class Kinematics:
    name: str
    dataset: str            # "tii" | "uzh"
    vehicle_class: str      # "tii_racing" | "uzh_snapdragon"
    t: np.ndarray           # (N,) seconds, native truth clock
    pos: np.ndarray         # (N, 3)
    v: np.ndarray           # (N, 3)
    a: np.ndarray           # (N, 3)
    j: np.ndarray           # (N, 3)
    yaw: np.ndarray         # (N,) model/path heading (may be time-shifted)
    yaw_truth: np.ndarray   # (N,) truth heading at t, the lean frame
    z: np.ndarray           # (N, 3) actual body z (thrust axis), unit
    lean: np.ndarray        # (N, 2) actual yaw-aligned lean
    valid: np.ndarray       # (N,) bool, interior of the SG window
    dt: float
    window_samples: int
    window_s: float

    def __len__(self) -> int:
        return len(self.t)

    def subset_valid(self) -> "Kinematics":
        m = self.valid
        return Kinematics(self.name, self.dataset, self.vehicle_class, self.t[m],
                          self.pos[m], self.v[m], self.a[m], self.j[m], self.yaw[m],
                          self.yaw_truth[m], self.z[m], self.lean[m], self.valid[m],
                          self.dt, self.window_samples, self.window_s)


def _odd_window(n: int, dt: float, window_s: float, poly: int) -> int:
    w = int(round(window_s / dt))
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 2)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w


def _kinematics_from_arrays(name: str, dataset: str, vehicle_class: str,
                            t: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray,
                            window_s: float, poly: int = 3) -> Kinematics:
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    quat = np.asarray(quat_xyzw, float)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = _odd_window(n, dt, window_s, poly)
    kw = dict(window_length=w, polyorder=poly, delta=dt, axis=0, mode="interp")
    v = savgol_filter(pos, deriv=1, **kw)
    a = savgol_filter(pos, deriv=2, **kw)
    j = savgol_filter(pos, deriv=3, **kw)
    R = Rotation.from_quat(quat).as_matrix()
    z = R[:, :, 2]
    yaw = Rotation.from_matrix(R).as_euler("ZYX")[:, 0]
    lean = world_to_yaw_frame(z[:, :2], yaw)
    valid = np.zeros(n, bool)
    half = w // 2
    valid[half:n - half] = True
    return Kinematics(name, dataset, vehicle_class, t, pos, v, a, j, yaw, yaw, z, lean,
                      valid, dt, w, w * dt)


def load_tii_kinematics(flight: str, window_s: float = 0.37, poly: int = 3,
                        root: Path | None = None) -> Kinematics:
    """One TII flight on the native 274.73 Hz mocap clock."""
    root = root or ROOT
    control, _ = flight_parts(flight)
    clip = load_tii(root / f"data/raw/tii/{control}/mocap_{flight}.csv", name=flight)
    return _kinematics_from_arrays(flight, "tii", "tii_racing", clip.t, clip.pos,
                                   clip.quat, window_s, poly)


def _uzh_arrays(seq: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth position, attitude and time on the 500 Hz Leica grid, within the image span."""
    truth = np.load(root / f"data/uzh/{seq}/truth.npz")
    gt = np.asarray(truth["gt"], float)          # timestamp tx ty tz qx qy qz qw
    t0 = float(truth["gt_t0"])
    t_rel = gt[:, 0] - t0
    frames = np.asarray(truth["t"], float)
    keep = (t_rel >= frames.min() - 1e-9) & (t_rel <= frames.max() + 1e-9)
    gt = gt[keep]
    t = gt[:, 0] - gt[0, 0]
    pos = gt[:, 1:4]
    quat = gt[:, 4:8]                            # xyzw, body -> world
    return t, pos, quat


def load_uzh_kinematics(seq: str, window_s: float = 0.10, poly: int = 3,
                        root: Path | None = None) -> Kinematics:
    """One UZH-FPV sequence on the truncated 500 Hz Leica clock."""
    root = root or ROOT
    t, pos, quat = _uzh_arrays(seq, root)
    return _kinematics_from_arrays(seq, "uzh", "uzh_snapdragon", t, pos, quat,
                                   window_s, poly)


def load_all(dataset: str, rig: str = "forward", window_s: float | None = None,
             root: Path | None = None) -> list[Kinematics]:
    if dataset == "tii":
        return [load_tii_kinematics(f, 0.37 if window_s is None else window_s, root=root)
                for f in TII_FLIGHTS]
    if dataset == "uzh":
        seqs = UZH_FORWARD if rig == "forward" else UZH_45
        return [load_uzh_kinematics(s, 0.10 if window_s is None else window_s, root=root)
                for s in seqs]
    raise ValueError(dataset)
