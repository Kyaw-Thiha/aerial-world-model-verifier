"""TII video: frame timestamps, introspected frames and camera calibration.

Additive companion to :mod:`src.datasets.tii` (which stays mocap-only). Builds a
video-rate :class:`Clip` whose ``t`` are frame timestamps on the mocap clock,
``pos``/``quat`` are mocap interpolated at those timestamps, and ``frame_paths``
point at JPEG members inside ``camera_<flight>.zip`` (``"<zip>!<member>"``).

Verified facts (2026-11-08, release v3.0.0):

* ``csv_raw/camera_<flight>.csv`` has columns ``timestamp`` (Unix microseconds,
  same clock as mocap) and ``filename``; one row per JPEG in the camera zip.
* ``<flight>_cam_ts_sync.csv`` is the authors' own interpolation of every mocap,
  IMU and thrust channel onto those frame timestamps. Its ``timestamp`` and
  ``img_filename`` columns are byte-identical to ``camera_<flight>.csv``, so the
  video-to-mocap offset is **0 by construction**, up to the sampling instant the
  hardware stamped (see ``meta["time_sync"]``). This loader does its own
  interpolation from the raw mocap so a lag can be applied later.
* Frame interval is 8.3 ms median (about 120.4 Hz measured, 120 Hz in metadata)
  with occasional 40 ms gaps from dropped frames; the raw timestamps are kept.
* Calibration: ``cv2.calibrateCamera`` output (``mtx`` 3x3, ``dist`` with
  k1,k2,p1,p2,k3), one file per camera group. fx != fy by 33%, which is a real
  property of the released calibration and is recorded, not corrected.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.datasets.clip import Clip
from src.datasets.tii import load_tii

# camera group -> calibration file in the official release
CALIB_FILES = {
    ("autonomous", "trackRATM"): "calib_a-trackRATM.json",
    ("autonomous", "ellipse"): "calib_ap-ellipse-lemniscate.json",
    ("autonomous", "lemniscate"): "calib_ap-ellipse-lemniscate.json",
    ("piloted", "ellipse"): "calib_ap-ellipse-lemniscate.json",
    ("piloted", "lemniscate"): "calib_ap-ellipse-lemniscate.json",
    ("piloted", "trackRATM"): "calib_p-trackRATM.json",
}

# drone_to_camera.json keys: three autonomous mounts plus one 30 deg piloted mount
EXTRINSIC_KEYS = {("autonomous", "ellipse"): "ellipse",
                 ("autonomous", "lemniscate"): "lemniscate",
                 ("autonomous", "trackRATM"): "trackRATM",
                 ("piloted", "ellipse"): "piloted",
                 ("piloted", "lemniscate"): "piloted",
                 ("piloted", "trackRATM"): "piloted"}


def flight_parts(flight: str) -> tuple[str, str]:
    """``flight-01a-ellipse`` -> ``("autonomous", "ellipse")``."""
    control = "autonomous" if flight.split("-")[1].endswith("a") else "piloted"
    track = flight.split("-", 2)[2]
    return control, track


def load_calibration(calib_dir: str | Path, control: str, track: str) -> dict:
    """Official OpenCV calibration for one camera group."""
    calib_dir = Path(calib_dir)
    path = calib_dir / CALIB_FILES[(control, track)]
    c = json.loads(path.read_text())
    K = np.asarray(c["mtx"], dtype=np.float64)
    dist = np.asarray(c["dist"], dtype=np.float64).ravel()
    if dist.size != 5:
        raise ValueError(f"{path}: expected 5 distortion coefficients, got {dist.size}")
    return {"model": "opencv_pinhole_k3", "K": K, "dist": dist,
            "source": str(path), "fx_fy_ratio": float(K[0, 0] / K[1, 1])}


def camera_to_body(calib_dir: str | Path, control: str, track: str) -> dict:
    """Camera-to-body rotation from ``drone_to_camera.json``.

    The JSON stores ``rotation`` per camera group as a quaternion (w, x, y, z) and
    documents it as "drone to camera ... includes the FLU to RDF conversion".
    Which matrix direction the quaternion actually denotes is decided empirically
    by :func:`elevation_check` (see ``docs/track_a_report.md``); this function
    returns **both** candidates so nothing is silently assumed.
    """
    path = Path(calib_dir) / "drone_to_camera.json"
    j = json.loads(path.read_text())
    q = j["rotation"][EXTRINSIC_KEYS[(control, track)]]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    return {"R_json": R, "R_json_T": R.T, "key": EXTRINSIC_KEYS[(control, track)],
            "quat_order_json": "wxyz", "source": str(path),
            "translation_m": [j["translation"]["x"], j["translation"]["y"],
                              j["translation"]["z"]]}


def elevation_check(R_cam_to_body: np.ndarray, R_body_to_world: np.ndarray,
                    camera_z_in_body: np.ndarray | None = None) -> np.ndarray:
    """Elevation of the camera optical axis above the world horizontal, degrees.

    Pure attitude check, no pixels: for every frame, take the optical axis
    (camera +z) into the body frame and then into the world frame, and report
    ``asin(world_z)``. A level, hovering flight should return the mount uptilt.
    """
    z = np.array([0.0, 0.0, 1.0]) if camera_z_in_body is None else np.asarray(camera_z_in_body)
    axis_body = R_cam_to_body.T @ z            # camera -> body
    axis_world = np.einsum("nij,j->ni", R_body_to_world, axis_body)
    return np.degrees(np.arcsin(np.clip(axis_world[:, 2], -1.0, 1.0)))


def frame_table(root: str | Path, flight: str) -> pd.DataFrame:
    """Frame timestamp/filename rows plus the matching zip member path."""
    control, _ = flight_parts(flight)
    root = Path(root)
    d = root / control
    cam = pd.read_csv(d / f"camera_{flight}.csv")
    if not {"timestamp", "filename"} <= set(cam.columns):
        raise ValueError(f"{d}/camera_{flight}.csv: unexpected columns {list(cam.columns)}")
    cam = cam.sort_values("timestamp").reset_index(drop=True)
    if not (np.diff(cam["timestamp"].to_numpy()) > 0).all():
        raise ValueError(f"{flight}: camera timestamps not strictly increasing")

    sync_path = d / f"{flight}_cam_ts_sync.csv"
    if sync_path.exists():
        sync = pd.read_csv(sync_path, usecols=["timestamp", "img_filename"])
        if len(sync) != len(cam) or not np.array_equal(sync["timestamp"].to_numpy(),
                                                       cam["timestamp"].to_numpy()):
            raise ValueError(f"{flight}: camera csv and cam_ts_sync disagree on timestamps")
        if list(sync["img_filename"]) != list(cam["filename"]):
            raise ValueError(f"{flight}: camera csv and cam_ts_sync disagree on filenames")

    zip_path = d / f"camera_{flight}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            have = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
        missing = [f for f in cam["filename"] if f not in have]
        if missing:
            raise ValueError(f"{flight}: {len(missing)} frames missing from {zip_path.name}, "
                             f"first {missing[0]}")
        cam["member"] = [have[f] for f in cam["filename"]]
        cam["zip"] = str(zip_path)
    else:
        cam["member"] = None
        cam["zip"] = str(zip_path)
    return cam


def load_tii_video(root: str | Path, flight: str, mocap_dir: str | Path = "data/raw/tii",
                   calib_dir: str | Path = "data/raw/tii_video/calibration",
                   fps: float | None = None, stride: int = 1,
                   lag_s: float = 0.0) -> Clip:
    """Video-rate clip for one flight.

    Parameters
    ----------
    root : directory holding ``<control>/camera_<flight>.zip`` etc.
    fps : declared frame rate. ``None`` measures it from the timestamps; the
        measured value always goes to ``meta["measured_fps"]``. Pass the dataset
        value (120) when the consumer needs a declared number rather than a
        measurement.
    stride : keep every ``stride``-th frame (1 = all frames).
    lag_s : shift applied to the mocap interpolation time base, i.e. truth is
        sampled at ``t_frame + lag_s``. Default 0; a non-zero value is recorded.
    """
    control, track = flight_parts(flight)
    mocap_path = Path(mocap_dir) / control / f"mocap_{flight}.csv"
    clip = load_tii(mocap_path, name=flight)
    cam = frame_table(root, flight)

    ts_us = cam["timestamp"].to_numpy(dtype=np.int64)
    t_rel = (ts_us - int(clip.meta["epoch_start_us"])).astype(np.float64) * 1e-6
    dts = np.diff(t_rel)
    measured_fps = 1.0 / float(np.median(dts))

    tq = t_rel + lag_s
    inside = (tq >= clip.t[0]) & (tq <= clip.t[-1])
    tt = np.clip(tq, clip.t[0], clip.t[-1])
    pos = np.column_stack([np.interp(tt, clip.t, clip.pos[:, k]) for k in range(3)])
    rots = Rotation.from_quat(clip.quat)
    rv = rots.as_rotvec()
    rv_t = np.column_stack([np.interp(tt, clip.t, rv[:, k]) for k in range(3)])
    quat = Rotation.from_rotvec(rv_t).as_quat()  # xyzw

    take = slice(None, None, stride)
    cal = load_calibration(calib_dir, control, track)
    ext = camera_to_body(calib_dir, control, track)
    intr = {
        "model": cal["model"], "K": cal["K"].tolist(), "dist": cal["dist"].tolist(),
        "image_size": [int(clip.meta.get("camera_metadata", {}).get("image_width", 640)),
                       int(clip.meta.get("camera_metadata", {}).get("image_height", 480))],
        "declared_fov_diagonal_deg": clip.meta.get("camera_metadata", {}).get("fov_diagonal"),
        "calibration_source": cal["source"],
        "camera_to_body": {"key": ext["key"], "R_json": ext["R_json"].tolist(),
                           "quat_order_json": ext["quat_order_json"],
                           "translation_m": ext["translation_m"],
                           "source": ext["source"]},
    }
    if stride != 1:
        intr["stride"] = stride
    missing_px = cam["member"].isna().any()
    paths = [None if m is None else f"{z}!{m}" for z, m in zip(cam["zip"], cam["member"])]

    out = Clip(
        name=flight, t=t_rel[take], pos=pos[take], quat=quat[take],
        quat_order="xyzw", fps=float(fps if fps is not None else measured_fps),
        cam_uptilt_deg=clip.cam_uptilt_deg,
        frame_paths=paths[take], intrinsics=intr,
        meta={"source_mocap": str(mocap_path), "source_video_root": str(root),
              "measured_fps": measured_fps,
              "declared_fps_metadata": clip.meta.get("camera_metadata", {}).get("fps"),
              "dt_median_s": float(np.median(dts)), "dt_min_s": float(dts.min()),
              "dt_max_s": float(dts.max()), "n_frames": int(len(cam)),
              "stride": stride, "n_used": int(len(t_rel[take])),
              "frames_zip_present": not missing_px,
              "exposure": clip.meta.get("camera_metadata", {}).get("exposure"),
              "brightness": clip.meta.get("light", {}).get("brightness"),
              "track": track, "control": control,
              "time_sync": {"method": "shared hardware clock: camera timestamps and mocap "
                                      "timestamps are both Unix microseconds in the release",
                            "lag_s_applied": lag_s,
                            "offset_uncertainty_s": 0.5 * float(np.median(dts)),
                            "note": "the stamp is the trigger instant; exposure latency is "
                                    "not documented and is not modelled"},
              "mocap_coverage_fraction": float(inside.mean())})
    return out


def load_frame(path: str) -> np.ndarray:
    """Read one JPEG either from ``<zip>!<member>`` or from disk. Returns BGR."""
    import cv2

    if "!" in path:
        zp, member = path.split("!", 1)
        with zipfile.ZipFile(zp) as z:
            buf = np.frombuffer(z.read(member), dtype=np.uint8)
    else:
        buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not decode {path}")
    return img


__all__ = ["CALIB_FILES", "flight_parts", "load_calibration", "camera_to_body",
           "elevation_check", "frame_table", "load_tii_video", "load_frame", "replace"]
