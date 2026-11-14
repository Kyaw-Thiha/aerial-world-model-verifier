"""UZH-FPV as a depicted-tilt testbed: rig table, published mount, truth pairing, gravity.

Facts this module encodes, each established in `reports/E01` (Task 1 audit):

1. **Ground-truth frame.** The ICRA19 paper (Sec. III-E) estimates the state (q, p, v) of the
   *Snapdragon IMU frame* in the Leica world frame, plus the prism-to-IMU translation. So the
   released quaternion is IMU-body -> Leica-world. `experiments/e_uzh_audit.py` checks that
   empirically (rest segment level, gravity along world z) rather than trusting the note.
2. **The mount is published.** Each of the four rigs ships a Kalibr `camchain-imucam` YAML
   with `T_cam_imu` (cam0 = `snappy_cam/stereo_l`, the camera the sequence images come from).
   `T_cam_imu` maps IMU coordinates to camera coordinates, so the camera->body rotation the
   rest of the project expects is `R_cb = R(T_cam_imu)^T`.
3. **The image clock is not always the truth clock.** `remote/e_sync.py` measures the offset
   from IMU-vs-truth angular speed. Sequences whose clocks coincide need no shift; the
   indoor-45 rig does (its images are stamped about 18 years away, but the numeric ranges
   overlap, so a naive interval filter pairs frames with unrelated truth).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------- rig table
# sequence -> rig. Rigs differ in mount, so they are never pooled for a mount fit.
SEQUENCE_RIG = {
    "indoor_forward_3": "indoor_forward",
    "indoor_forward_5": "indoor_forward",
    "indoor_forward_6": "indoor_forward",
    "indoor_forward_7": "indoor_forward",
    "indoor_forward_9": "indoor_forward",
    "indoor_forward_10": "indoor_forward",
    "outdoor_forward_1": "outdoor_forward",
    "outdoor_forward_3": "outdoor_forward",
    "outdoor_forward_5": "outdoor_forward",
    "indoor_45_2": "indoor_45",
    "indoor_45_4": "indoor_45",
    "indoor_45_9": "indoor_45",
    "indoor_45_13": "indoor_45",
    "outdoor_45_1": "outdoor_45",
}

SEQUENCE_ENV = {s: ("outdoor" if s.startswith("outdoor") else "indoor") for s in SEQUENCE_RIG}

# Pre-registered split (configs/e_prereg.yaml).
FIT_SEQUENCES = ("indoor_forward_3", "indoor_forward_5", "indoor_45_2", "outdoor_forward_1")


def rig_of(seq: str) -> str:
    return SEQUENCE_RIG[seq]


def is_forward(seq: str) -> bool:
    return rig_of(seq).endswith("forward")


def is_indoor(seq: str) -> bool:
    return SEQUENCE_ENV[seq] == "indoor"


def all_sequences() -> list[str]:
    return list(SEQUENCE_RIG)


# --------------------------------------------------------------------------- mount
def published_mount(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R_cb, t_cam_imu) for the camera the frames come from, from the rig's Kalibr file.

    ``R_cb`` is camera->body (the convention of ``src.perception.frames``), i.e. the
    transpose of the rotation inside ``T_cam_imu``.
    """
    cam0 = meta["kalibr_imu_cam0"]
    T = np.asarray(cam0["T_cam_imu"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_imu is {T.shape}, expected 4x4")
    R_ci = T[:3, :3]
    if not np.allclose(R_ci @ R_ci.T, np.eye(3), atol=1e-6):
        raise ValueError("T_cam_imu rotation is not orthonormal")
    return R_ci.T, T[:3, 3]


def optical_axis_in_body(R_cb: np.ndarray) -> np.ndarray:
    """Camera +z (optical axis) in body coordinates = third column of R_cb."""
    return np.asarray(R_cb, dtype=np.float64)[:, 2]


def optical_axis_elevation_deg(R_cb: np.ndarray) -> float:
    """Angle of the optical axis below the body's horizontal plane, degrees (down positive)."""
    z = optical_axis_in_body(R_cb)
    return float(np.degrees(np.arcsin(np.clip(-z[2], -1.0, 1.0))))


# --------------------------------------------------------------------------- truth
@dataclass
class UzhTruth:
    """One sequence's truth, already paired to the frames."""

    seq: str
    rig: str
    env: str
    t: np.ndarray            # (N,) seconds relative to the first kept frame
    t_unix: np.ndarray       # (N,) image timestamps on the image clock
    pos: np.ndarray          # (N, 3) m, Leica world frame
    quat: np.ndarray         # (N, 4) xyzw, IMU body -> world
    R_bw: np.ndarray         # (N, 3, 3)
    offset_s: float          # gt_time = img_time + offset  (clock correction applied)
    offset_corr: float | None
    n_frames: int
    meta: dict

    @property
    def up_body(self) -> np.ndarray:
        """World up in body (IMU) coordinates, (N, 3)."""
        return np.einsum("nji,j->ni", self.R_bw, np.array([0.0, 0.0, 1.0]))

    @property
    def tilt_deg(self) -> np.ndarray:
        """Angle between body z and world up, degrees."""
        return np.degrees(np.arccos(np.clip(self.up_body[:, 2], -1.0, 1.0)))

    def R_cb(self) -> np.ndarray:
        return published_mount(self.meta)[0]

    def up_cam_truth(self) -> np.ndarray:
        """World up in camera coordinates from truth attitude and the published mount."""
        return np.einsum("ij,nj->ni", self.R_cb(), self.up_body)

    def up_cam_truth_for(self, R_cb: np.ndarray) -> np.ndarray:
        return np.einsum("ij,nj->ni", np.asarray(R_cb, dtype=np.float64), self.up_body)

    def yaw(self) -> np.ndarray:
        return Rotation.from_matrix(self.R_bw).as_euler("ZYX")[:, 0]


def _read_rows(text: str, min_cols: int = 8) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= min_cols:
            rows.append(v[:8])
    out = np.asarray(rows, dtype=np.float64)
    return out[np.argsort(out[:, 0])]


def nearest_index(query: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the nearest sample of the sorted ``grid`` for every query time."""
    idx = np.searchsorted(grid, query).clip(1, len(grid) - 1)
    left = np.abs(grid[idx - 1] - query) < np.abs(grid[idx] - query)
    return np.where(left, idx - 1, idx)


def load_truth(root: str | Path, seq: str, offset: float | None = None,
               rig: str | None = None, env: str | None = None) -> UzhTruth:
    """Load one prepared sequence.

    Pairing happens once, in ``remote/e_prepare.py``, using the measured clock offset; this
    loader only checks that the offset it is told about is the offset that was applied, so a
    sequence prepared on the wrong clock cannot be analysed silently.
    """
    root = Path(root)
    d = root / seq
    meta = json.loads((d / "meta.json").read_text())
    applied = float(meta.get("clock_offset_applied_s", 0.0))
    if offset is not None and abs(float(offset) - applied) > 1e-9:
        raise ValueError(f"{seq}: prepared with offset {applied!r} s, asked for {offset!r} s")
    truth = np.load(d / "truth.npz")
    t = np.asarray(truth["t"], dtype=np.float64)
    pos = np.asarray(truth["pos"], dtype=np.float64)
    quat = np.asarray(truth["quat"], dtype=np.float64)
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{seq}: frame times are not strictly increasing")
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    R_bw = Rotation.from_quat(quat).as_matrix()
    t_unix = np.asarray(meta["t_unix"], dtype=np.float64)
    if len(t_unix) != len(t):
        raise ValueError(f"{seq}: {len(t_unix)} image timestamps for {len(t)} poses")
    sync = (load_sync(root).get(seq) or {})
    rig = rig or SEQUENCE_RIG.get(seq)
    env = env or SEQUENCE_ENV.get(seq) or meta.get("environment")
    if rig is None:
        raise KeyError(f"{seq}: not in the rig table; pass rig= and env= explicitly")
    return UzhTruth(seq=seq, rig=rig, env=env, t=t, t_unix=t_unix,
                    pos=pos, quat=quat, R_bw=R_bw, offset_s=applied,
                    offset_corr=sync.get("offset_peak_corr"), n_frames=int(len(t)), meta=meta)


def load_sync(root: str | Path) -> dict[str, dict]:
    """Per-sequence clock-offset records written by ``remote/e_sync.py``."""
    root = Path(root)
    out = {}
    for p in sorted(root.glob("sync/*.json")):
        rec = json.loads(p.read_text())
        out[rec["seq"]] = rec
    return out


def frame_paths(root: str | Path, seq: str, kind: str = "frames_pinhole",
                n: int | None = None) -> list[Path]:
    d = Path(root) / seq / kind
    files = sorted(d.glob("*.png"))
    return files[:n] if n is not None else files


# --------------------------------------------------------------------------- kinematics
def dense_kinematics(gt: np.ndarray, window_s: float = 0.10, poly: int = 3) -> dict:
    """Velocity and acceleration of the dense truth table on its own grid."""
    from scipy.signal import savgol_filter

    t = gt[:, 0]
    p = gt[:, 1:4]
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = max(5, int(round(window_s / dt)))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 else n - 1)
    v = np.column_stack([savgol_filter(p[:, k], w, poly, deriv=1, delta=dt) for k in range(3)])
    a = np.column_stack([savgol_filter(p[:, k], w, poly, deriv=2, delta=dt) for k in range(3)])
    return {"t": t, "p": p, "v": v, "a": a, "window_s": w * dt, "samples": w}


def required_tilt_deg(a: np.ndarray, v: np.ndarray, k_d: float = 0.0, g: float = 9.81):
    """Tilt of the thrust axis implied by the path (TII's flatness equation)."""
    u = np.asarray(a, dtype=np.float64) + np.array([0.0, 0.0, g]) + k_d * np.asarray(v)
    n = np.linalg.norm(u, axis=1)
    ok = n > 1e-6
    u = np.where(ok[:, None], u / np.where(ok, n, 1.0)[:, None], np.nan)
    return np.degrees(np.arccos(np.clip(u[:, 2], -1.0, 1.0))), u, ok
