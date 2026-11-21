"""Clip construction for the simulated labelled set (Track I, Step 2).

A clip is fully determined by one row of ``configs/sim_splits_frozen.json``: the frozen row
gives the scene, the maneuver and its parameters, the mount, the defect, its magnitude and the
interval seed, and everything else is closed form (``src/sim/flight.py``) or construction
(``src/sim/labels.py``). Nothing is drawn at build time that is not either in the row or
derived deterministically from it, which is what makes the manifest a description of the data
rather than a log of a run.

The clip carries three levels of truth:

* **commanded** attitude and required lean/tilt -- what the path demands (flatness);
* **rendered** camera attitude and depicted lean/tilt -- what the pixels will show, exact
  because the renderer is *given* the pose;
* **labels** -- feasible/infeasible, defect family, magnitude in degrees, exact interval.

The last two are what a verifier is scored against; the first is what a verifier must infer
from the path alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.frames import camera_to_body_pitched
from src.sim.flight import Flight, build
from src.sim.labels import (Defect, camera_attitude, defect_from_row, depicted_gt,
                            realised_magnitude, required_gt)

E3 = np.array([0.0, 0.0, 1.0])


@dataclass
class Clip:
    row: dict
    flight: Flight
    defect: Defect | None
    R_cb: np.ndarray
    R_bw_cam: np.ndarray
    lean_req: np.ndarray
    lean_dep: np.ndarray
    tilt_req: np.ndarray
    tilt_dep: np.ndarray
    active: np.ndarray
    labels: dict
    frame_t: np.ndarray | None = None
    frame_idx: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- render side ----------------------------------------------------------
    def poses(self) -> np.ndarray:
        """(M,4,4) camera-to-world poses at the video rate, OpenCV RDF."""
        R_wc = self.R_bw_cam @ self.R_cb
        M = len(self.frame_idx)
        out = np.tile(np.eye(4), (M, 1, 1))
        out[:, :3, :3] = R_wc[self.frame_idx]
        out[:, :3, 3] = self.flight.pos[self.frame_idx]
        return out

    # -- storage --------------------------------------------------------------
    def gt_arrays(self) -> dict:
        f = self.flight
        return {"t": f.t, "pos": f.pos, "vel": f.vel, "acc": f.acc, "yaw": f.yaw,
                "R_bw": f.R_bw, "R_bw_cam": self.R_bw_cam, "R_cb": self.R_cb,
                "lean_req": self.lean_req, "lean_dep": self.lean_dep,
                "tilt_req": self.tilt_req, "tilt_dep": self.tilt_dep,
                "active": self.active.astype(np.int8),
                "gt_hz": np.array([f.gt_hz]), "k_d": np.array([f.k_d])}

    def render_arrays(self) -> dict:
        """The bundle the Blender adapter consumes (same format as the spike's jobs)."""
        idx = self.frame_idx
        return {"poses": self.poses(), "yaw": self.flight.yaw[idx],
                "tilt_gt_deg": self.tilt_dep[idx], "lean_gt": self.lean_dep[idx],
                "lean_req": self.lean_req[idx], "R_cb": self.R_cb,
                "uptilt_deg": np.array([self.row["uptilt_deg"]]),
                "gt_hz": np.array([self.flight.gt_hz]), "k_d": np.array([self.flight.k_d]),
                "frame_t": self.frame_t}

    def video_rows(self) -> list[dict]:
        """One row per video frame: what the truth and the defect are at that instant."""
        rows = []
        for k, (t, i) in enumerate(zip(self.frame_t, self.frame_idx)):
            rows.append({"clip_id": self.row["clip_id"], "frame": k,
                         "t": float(t),
                         "lean_req_x": float(self.lean_req[i, 0]),
                         "lean_req_y": float(self.lean_req[i, 1]),
                         "lean_dep_x": float(self.lean_dep[i, 0]),
                         "lean_dep_y": float(self.lean_dep[i, 1]),
                         "tilt_req_deg": float(self.tilt_req[i]),
                         "tilt_dep_deg": float(self.tilt_dep[i]),
                         "defect_active": int(bool(self.active[i]))})
        return rows


def _mount(uptilt_deg: float, mount_yaw_deg: float) -> np.ndarray:
    R = camera_to_body_pitched(np.radians(float(uptilt_deg)))
    if abs(mount_yaw_deg) > 1e-12:
        c, s = np.cos(np.radians(mount_yaw_deg)), np.sin(np.radians(mount_yaw_deg))
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R
    return R


def build_clip(row: dict) -> Clip:
    """Resolve one frozen row into truth, labels and the render schedule."""
    flight: Flight = build(row["maneuver"], **row["params"])
    flight.video_fps = 30.0
    duration = float(flight.t[-1])
    defect = defect_from_row(row, duration)
    R_cb = _mount(row["uptilt_deg"], row.get("mount_yaw_deg", 0.0))
    R_bw_cam = camera_attitude(flight, defect)
    lean_req, tilt_req = required_gt(flight)
    lean_dep, tilt_dep = depicted_gt(R_bw_cam, R_cb)
    active = defect.active(flight.t) if defect is not None else np.ones_like(
        flight.t, dtype=bool)

    # render schedule: the uniform 30 fps grid over the flight
    m = int(np.floor(duration * 30.0 + 1e-9)) + 1
    frame_t = np.arange(m) / 30.0
    frame_idx = np.round(frame_t * flight.gt_hz).astype(int).clip(0, len(flight.t) - 1)

    labels = _labels(row, flight, defect, tilt_req, active, duration)
    return Clip(row=row, flight=flight, defect=defect, R_cb=R_cb, R_bw_cam=R_bw_cam,
                lean_req=lean_req, lean_dep=lean_dep, tilt_req=tilt_req, tilt_dep=tilt_dep,
                active=active, labels=labels, frame_t=frame_t, frame_idx=frame_idx)


def _labels(row: dict, flight: Flight, defect: Defect | None, tilt_req: np.ndarray,
            active: np.ndarray, duration: float) -> dict:
    fam = row.get("defect_family", "feasible")
    out = {
        "clip_id": row["clip_id"],
        "split": row["split"],
        "family": row["family"],
        "scene_seed": int(row["scene_seed"]),
        "maneuver": row["maneuver"],
        "uptilt_deg": float(row["uptilt_deg"]),
        "duration_s": duration,
        "n_frames": int(np.floor(duration * 30.0 + 1e-9)) + 1,
        "n_gt": int(len(flight.t)),
        "peak_tilt_req_deg": float(tilt_req.max()),
        "median_tilt_req_deg": float(np.median(tilt_req)),
        "peak_thrust_ratio": float((1.0 / np.cos(np.radians(tilt_req))).max()),
    }
    if fam == "thrust_limit":
        out.update({"label": "infeasible", "violation": "thrust_limit",
                    "defect_family": "thrust_limit",
                    "magnitude_deg": float(tilt_req.max() - 60.0),
                    "magnitude_unit": "excess peak tilt over the 60 deg / 2.0 g limit (deg)",
                    "interval_start_s": 0.0, "interval_end_s": duration,
                    "interval_fraction": 1.0,
                    "diagnostic": f"peak thrust ratio {out['peak_thrust_ratio']:.2f} g"})
        return out
    if defect is None:
        out.update({"label": "feasible", "violation": None, "defect_family": "feasible",
                    "magnitude_deg": 0.0, "magnitude_unit": "none",
                    "interval_start_s": None, "interval_end_s": None,
                    "interval_fraction": 0.0})
        return out
    mag = realised_magnitude(defect, tilt_req, active)
    t0, t1 = defect.interval_s if defect.interval_s is not None else (0.0, duration)
    out.update({"label": "infeasible", "violation": defect.kind,
                "defect_family": fam,
                "magnitude_deg": mag["magnitude_deg"],
                "magnitude_unit": mag["magnitude_unit"],
                "interval_start_s": float(t0), "interval_end_s": float(t1),
                "interval_fraction": float(defect.fraction),
                "defect_magnitude": float(defect.magnitude)})
    return out


def manifest_row(clip: Clip) -> dict:
    """The row the manifest and ``docs/DATA.md`` publish (sha256 fields filled on upload)."""
    r = dict(clip.labels)
    r["params_json"] = _json(clip.row["params"])
    r["renderer"] = "blender"
    r["engine"] = "EEVEE"
    r["samples"] = 16
    r["r2_video"] = ""
    r["r2_gt"] = ""
    r["sha256_video"] = ""
    r["sha256_gt"] = ""
    return r


def _json(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
