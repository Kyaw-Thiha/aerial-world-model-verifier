"""Shared data plumbing for Track P (rate consistency and thrust envelope).

Every loader returns the same dictionary:

* ``kind``      -- ``sim``, ``uzh`` or ``tii``
* ``unit``      -- clip id or sequence/flight name
* ``t``, ``pos``, ``yaw`` -- truth path (native rate) and the path yaw where it exists
* ``R_bw``, ``R_bw_cam``, ``R_cb`` -- body attitude the path requires, body attitude actually
  rendered (differs only on defect clips), and the camera mount
* ``t_video``, ``R_est`` -- video frame times and the camera-to-world rotation from DPVO
  (arbitrary constant gauge; only differences are used)
* ``fps_video`` -- frame rate of the rotation estimate

Held out material (100 simulated clips, TII 13p and 07a) is filtered here and never opened.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.datasets.tii import load_tii
from src.datasets.tii_video import camera_to_body, flight_parts
from src.depicted.uzh import published_mount

ROOT = Path(__file__).resolve().parents[1]
SIM_GT = ROOT / "data/n05/gt"
SIM_SCORES = ROOT / "data/n05/scores"
SIM_POSES = ROOT / "data/n05/poses/sim200/scale1"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"
UZH = ROOT / "data/uzh"
UZH_POSES = ROOT / "data/n01/poses/uzh"
TII = ROOT / "data/raw/tii"
TII_CALIB = ROOT / "data/raw/tii_calib"
TII_POSES = ROOT / "data/raw/dpvo"
H04_PATHS = ROOT.parent / "awmv-h3/results/h04/paths/paths.npz"

UZH_OUTDOOR = ("outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5", "outdoor_45_1")
TII_HELDOUT = ("flight-13p-trackRATM", "flight-07a-lemniscate")
TII_DEV = ("flight-01a-ellipse", "flight-04a-ellipse", "flight-13a-trackRATM",
           "flight-01p-ellipse", "flight-07p-lemniscate", "flight-16p-trackRATM")


def sim_manifest(split: str = "dev") -> dict[str, dict]:
    rows = list(csv.DictReader(open(MANIFEST)))
    out = {r["clip_id"]: r for r in rows if r["split"] == split}
    assert len(out) == (200 if split == "dev" else 100)
    return out


def _sim(row: dict) -> dict:
    cid = row["clip_id"]
    g = np.load(SIM_GT / f"{cid}.npz")
    t = np.asarray(g["t"], float)
    pos = np.load(SIM_POSES / f"{cid}.npz")
    R_est = np.asarray(pos["c2w"], float)[:, :3, :3]
    t_video = np.asarray(pos["t"], float)
    hz = float(np.asarray(g["gt_hz"]).ravel()[0])
    idx = np.clip(np.round(t_video * hz).astype(int), 0, len(t) - 1)
    return {
        "kind": "sim", "unit": cid, "row": row,
        "t": t, "pos": np.asarray(g["pos"], float),
        "vel": np.asarray(g["vel"], float), "acc": np.asarray(g["acc"], float),
        "yaw": np.asarray(g["yaw"], float),
        "R_bw": np.asarray(g["R_bw"], float),
        "R_bw_cam": np.asarray(g["R_bw_cam"], float),
        "R_bw_v": np.asarray(g["R_bw"], float)[idx],
        "R_bw_cam_v": np.asarray(g["R_bw_cam"], float)[idx],
        "R_cb": np.asarray(g["R_cb"], float),
        "t_video": t_video,
        "fps_video": float(1.0 / np.median(np.diff(t_video))), "R_est": R_est,
        "duration_s": float(t[-1]),
    }


def _uzh(seq: str) -> dict | None:
    meta = json.loads((UZH / seq / "meta.json").read_text())
    tr = np.load(UZH / seq / "truth.npz")
    R_cb, _ = published_mount(meta)
    gt = np.asarray(tr["gt"], float)
    t0 = float(tr["gt_t0"])
    R_bw = Rotation.from_quat(gt[:, 4:8]).as_matrix()
    p = UZH_POSES / f"{seq}.npz"
    if not p.exists():
        return None
    P = np.load(p)
    R_est = np.asarray(P["c2w"], float)[:, :3, :3]
    t_video = np.asarray(tr["t_unix"], float) - t0
    n = min(len(R_est), len(t_video))
    return {
        "kind": "uzh", "unit": seq, "row": None,
        "t": gt[:, 0] - t0, "pos": gt[:, 1:4], "vel": None, "acc": None, "yaw": None,
        "R_bw": R_bw, "R_bw_cam": R_bw,
        "R_bw_v": R_bw[np.clip(np.searchsorted(gt[:, 0] - t0, t_video[:n]), 0, len(R_bw) - 1)],
        "R_bw_cam_v": R_bw[np.clip(np.searchsorted(gt[:, 0] - t0, t_video[:n]), 0,
                                   len(R_bw) - 1)],
        "R_cb": R_cb,
        "t_video": t_video[:n], "fps_video": float(meta["fps_measured"]),
        "R_est": R_est[:n], "duration_s": float(gt[-1, 0] - t0),
    }


def _tii(flight: str) -> dict | None:
    control, track = flight_parts(flight)
    path = TII / control / f"mocap_{flight}.csv"
    R_cb = camera_to_body(TII_CALIB, control, track)["R_json"]
    clip = load_tii(path)
    t = np.asarray(clip.t, float)
    d = np.load(TII_POSES / f"{flight}_stride8_fov100_s0.5/poses.npz")
    q = np.asarray(d["poses"], float)[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    R_est = Rotation.from_quat(q).as_matrix()
    ts = np.asarray(d["frame_ts_us"], float)
    t_video = (ts - ts[0]) * 1e-6
    tq = t - t[0]
    sl = Slerp(tq, Rotation.from_quat(np.asarray(clip.quat, float)))
    R_bw = sl(np.clip(t_video, tq[0], tq[-1])).as_matrix()
    return {
        "kind": "tii", "unit": flight, "row": None,
        "t": tq, "pos": np.asarray(clip.pos, float), "vel": None, "acc": None, "yaw": None,
        "R_bw": R_bw, "R_bw_cam": R_bw, "R_bw_v": R_bw, "R_bw_cam_v": R_bw, "R_cb": R_cb,
        "t_video": t_video, "fps_video": 1.0 / float(np.median(np.diff(t_video))),
        "R_est": R_est, "duration_s": float(tq[-1]),
    }


def iter_units(which: str) -> list[dict]:
    """``sim`` (200 dev clips), ``uzh`` (14) or ``tii`` (6 development flights)."""
    if which == "sim":
        return [_sim(r) for _, r in sorted(sim_manifest("dev").items())]
    if which == "uzh":
        out = [_uzh(s) for s in sorted(p.name for p in UZH.iterdir() if (p / "truth.npz").exists())]
        return [u for u in out if u is not None]
    if which == "tii":
        out = []
        for f in TII_DEV:
            assert f not in TII_HELDOUT
            u = _tii(f)
            if u is not None:
                out.append(u)
        return out
    raise ValueError(which)


# ------------------------------------------------------------------ attitude from a path
def attitude_from_axis_and_yaw(b3: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ZYX body-to-world attitude whose z axis is ``b3`` and whose heading is ``yaw``.

    Uses the project's lean convention ``l = (cos(phi) sin(theta), -sin(phi))`` for the
    horizontal part of the thrust axis in the yaw-aligned frame.
    """
    b3 = np.asarray(b3, float)
    yaw = np.asarray(yaw, float)
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = c * b3[:, 0] - s * b3[:, 1]
    ly = s * b3[:, 0] + c * b3[:, 1]
    roll = -np.arcsin(np.clip(ly, -1, 1))
    cosr = np.cos(roll)
    pitch = np.arcsin(np.clip(lx / np.where(np.abs(cosr) > 1e-9, cosr, np.nan), -1, 1))
    out = np.empty((len(b3), 3, 3))
    for i in range(len(b3)):
        R = (Rotation.from_euler("z", yaw[i]).as_matrix()
             @ Rotation.from_euler("y", pitch[i]).as_matrix()
             @ Rotation.from_euler("x", roll[i]).as_matrix())
        out[i] = R
    return out


def scale_lean(R_bw: np.ndarray, s: float) -> np.ndarray:
    """``roll -> s*roll``, ``pitch -> s*pitch`` in the ZYX decomposition (sim defect rule)."""
    e = Rotation.from_matrix(np.asarray(R_bw, float)).as_euler("ZYX")
    e2 = e.copy()
    e2[:, 1] *= s
    e2[:, 2] *= s
    return Rotation.from_euler("ZYX", e2).as_matrix()


def remove_tilt(R_bw: np.ndarray) -> np.ndarray:
    """Level camera keeping the path yaw (sim ``tilt_removed`` rule)."""
    e = Rotation.from_matrix(np.asarray(R_bw, float)).as_euler("ZYX")
    e2 = e.copy()
    e2[:, 1] = 0.0
    e2[:, 2] = 0.0
    return Rotation.from_euler("ZYX", e2).as_matrix()


def mirror_roll(R_bw: np.ndarray) -> np.ndarray:
    e = Rotation.from_matrix(np.asarray(R_bw, float)).as_euler("ZYX")
    e2 = e.copy()
    e2[:, 2] *= -1.0
    return Rotation.from_euler("ZYX", e2).as_matrix()
