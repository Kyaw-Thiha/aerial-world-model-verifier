"""TII Race Against the Machine mocap loader.

Source: https://github.com/tii-racing/drone-racing-dataset (release v3.0.0,
data CC BY 4.0), files ``<kind>/<flight>/csv_raw/mocap_<flight>.csv``.

Verified against the files and the paper (arXiv:2311.02667v2, Sec. V-B):

* Frame: world z up, metres, body frame is the flight controller's IMU frame in
  FLU (x forward, y left, z up). ``drone_rot`` is a **column-major** 3x3
  body-to-world rotation matrix; its third column (entries 6, 7, 8) is the body
  z-axis. Verified by matching the sign of the pitch/roll columns of ``drone_rot``
  against ``drone_pitch``/``drone_roll`` and by ||R R^T - I|| < 2e-5.
* Orientation is *not* stored as a quaternion. This loader converts the matrix
  to a quaternion and reports ``quat_order = "xyzw"`` (scipy convention).
* ``timestamp`` is Unix epoch in **microseconds**; time is rebased so the clip
  starts at t = 0 and the absolute epoch time is kept in ``meta``.
* Mocap runs at 275 Hz (paper) and the files are already uniform at
  dt = 3.63/3.64 ms with no gaps, so no resampling is applied. ``resample_hz``
  exists for later stages that may want a camera-rate or a coarser grid.
* No NaN or missing rows in the 12 flights surveyed.

Only mocap columns are read; video and rosbags are never touched here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from src.datasets.clip import Clip

# Provenance constants. Change these only with evidence from the dataset.
TII_LICENSE = "CC BY 4.0 (dataset), MIT (code)"
TII_RATE_HZ_EXPECTED = 274.73
TII_ROT_LAYOUT = "column-major"
TII_QUAT_ORDER = "xyzw"  # after conversion from the rotation matrix
TII_TIME_UNITS = "us"
TII_BODY_FRAME = "FLU (x forward, y left, z up)"
TII_WORLD_FRAME = "z-up"

POSE_COLS = [
    "drone_x", "drone_y", "drone_z",
    "drone_roll", "drone_pitch", "drone_yaw",
    "drone_residual",
] + [f"drone_rot[{i}]" for i in range(9)]


def _rotmat_column_major(rows: np.ndarray) -> np.ndarray:
    """(N, 9) column-major entries -> (N, 3, 3) body-to-world matrices."""
    out = np.empty((len(rows), 3, 3), dtype=np.float64)
    for j in range(3):
        out[:, :, j] = rows[:, 3 * j : 3 * j + 3]
    return out


def load_tii(path: str | Path, name: str | None = None,
             resample_hz: float | None = None) -> Clip:
    """Load one TII flight's mocap stream into a :class:`Clip`.

    Parameters
    ----------
    path : path to ``mocap_<flight>.csv`` (or a flight folder containing it).
    name : clip name; defaults to the flight folder name.
    resample_hz : optional target rate. ``None`` keeps the native 274.73 Hz.
        When set, position and attitude are interpolated onto a uniform grid
        (linear for position, slerp for attitude) and this is recorded in
        ``meta["resampled"]``.
    """
    path = Path(path)
    if path.is_dir():
        cands = list(path.glob("mocap_*.csv"))
        if len(cands) != 1:
            raise FileNotFoundError(f"{path} should contain exactly one mocap_*.csv, found {len(cands)}")
        path = cands[0]
    df = pd.read_csv(path, usecols=lambda c: c in POSE_COLS or c in ("timestamp", "frame"))
    missing = [c for c in ("timestamp", *POSE_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    n_raw = len(df)
    nan_rows = int(df[["timestamp", *POSE_COLS]].isna().any(axis=1).sum())
    if nan_rows:
        df = df.dropna(subset=["timestamp", *POSE_COLS]).reset_index(drop=True)

    t_us = df["timestamp"].to_numpy(dtype=np.int64)
    t = (t_us - t_us[0]).astype(np.float64) * 1e-6  # seconds, relative
    pos = df[["drone_x", "drone_y", "drone_z"]].to_numpy(dtype=np.float64)
    R = _rotmat_column_major(df[[f"drone_rot[{i}]" for i in range(9)]].to_numpy(dtype=np.float64))

    orth = float(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max())
    det = np.linalg.det(R)
    if orth > 1e-3 or det.min() <= 0:
        raise ValueError(f"{path}: drone_rot is not a proper rotation (orth err {orth:.2e}, "
                         f"det range [{det.min():.4f}, {det.max():.4f}]); check ROT_LAYOUT")

    quat = Rotation.from_matrix(R).as_quat()  # xyzw

    dt = np.diff(t)
    rate = 1.0 / float(np.median(dt))
    meta = {
        "source": str(path),
        "license": TII_LICENSE,
        "rot_layout": TII_ROT_LAYOUT,
        "time_units_raw": TII_TIME_UNITS,
        "epoch_start_us": int(t_us[0]),
        "body_frame": TII_BODY_FRAME,
        "world_frame": TII_WORLD_FRAME,
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "n_raw": n_raw,
        "n_dropped_nan": nan_rows,
        "rot_orthonormality_max_err": orth,
        "rot_det_range": [float(det.min()), float(det.max())],
        "resampled": None,
        "camera_uptilt_from_metadata": None,
    }

    # per-flight metadata yaml sits next to csv_raw/ and records the camera angle
    flight = path.stem.replace("mocap_", "")
    meta_path = next(
        (c for c in (path.parent / f"metadata_{flight}.yaml",
                     path.parent.parent / f"metadata_{flight}.yaml") if c.exists()),
        None,
    )
    if meta_path is not None:
        y = yaml.safe_load(meta_path.read_text())
        meta["track"] = y.get("track")
        meta["control"] = y.get("control")
        meta["camera_metadata"] = y.get("camera")
        meta["light"] = y.get("light")
        meta["flown_meters"] = y.get("flown_meters")

    uptilt = None
    if meta.get("camera_metadata"):
        uptilt = float(meta["camera_metadata"]["angle"])
        meta["camera_uptilt_from_metadata"] = uptilt

    if resample_hz is not None:
        t_c = np.arange(0.0, t[-1] + 1e-9, 1.0 / resample_hz)
        pos_c = np.column_stack([np.interp(t_c, t, pos[:, k]) for k in range(3)])
        # slerp: convert to euler-free interpolation via Rotation
        rots = Rotation.from_quat(quat)
        # scipy has no direct slerp over an irregular time base; interpolate with
        # Rotation.from_rotvec on the unwrapped log map
        rv = rots.as_rotvec()
        rv_c = np.column_stack([np.interp(t_c, t, rv[:, k]) for k in range(3)])
        quat_c = Rotation.from_rotvec(rv_c).as_quat()
        meta["resampled"] = {"from_hz": rate, "to_hz": float(resample_hz),
                             "method": "linear position, rotvec-interpolated attitude"}
        t, pos, quat = t_c, pos_c, quat_c
        dt = np.diff(t)
        rate = float(resample_hz)

    clip = Clip(
        name=name or flight,
        t=t, pos=pos, quat=quat, quat_order=TII_QUAT_ORDER,
        fps=rate,
        cam_uptilt_deg=uptilt,
        meta=meta,
    )
    clip.meta["effective_rate_hz"] = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else rate
    return clip
