"""Phase 3a: build the frame table, the clip table and the edit rotation specs.

Everything that needs mocap truth is computed here on CPU, once per flight:

* the stride-10 frame table (timestamps, member names) matching the Phase 1 GeoCalib run,
* the implied thrust axis from **mocap position** at the native rate, sampled at the
  frame times (Phase 0 physics: Savitzky-Golay window 101 samples = 0.37 s, poly 3,
  ``k_d = 0.295`` fixed — see configs/verdict_preregistration.yaml),
* the per-frame camera-frame rotation ``M`` for every edit condition,
* the 4 s clip table.

The GPU worker only decodes a frame, applies ``M`` (a remap) and runs GeoCalib, so the
physics and the clip definition never live in the remote image.

Output: data/processed/p3a/<tag>/<flight>.npz and data/processed/p3a/<tag>/clips.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

from src.datasets.tii import load_tii
from src.datasets.tii_video import flight_parts, frame_table, load_tii_video
from src.negatives.edits import (camera_frame_map, condition_maps, delta_from_attitudes,
                                 rotation_angle_deg, scaled_tilt_attitude)
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import G0, implied_thrust_axis, tilt_deg

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
CONDITIONS = ["identity", "tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"]
SG_WINDOW_S = 0.37
SG_POLY = 3
K_D = 0.295
CLIP_LEN_S = 4.0


def implied_axis_at(t_frame: np.ndarray, mocap_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Unit thrust axis implied by mocap position, and speed, sampled at ``t_frame``."""
    clip = load_tii(mocap_path, name=mocap_path.stem)
    dt = 1.0 / clip.fps
    w = min(101, len(clip.pos) - 1)
    if w % 2 == 0:
        w -= 1
    vel = savgol_filter(clip.pos, w, SG_POLY, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(clip.pos, w, SG_POLY, deriv=2, delta=dt, axis=0, mode="interp")
    axis = implied_thrust_axis(acc, vel, K_D, G0)
    axis[: w // 2] = np.nan
    axis[-(w // 2):] = np.nan
    tt = np.clip(t_frame, clip.t[0], clip.t[-1])
    out = np.column_stack([np.interp(tt, clip.t, axis[:, k]) for k in range(3)])
    speed = np.interp(tt, clip.t, np.linalg.norm(vel, axis=1))
    return out, speed


def frame_speed(t_frame: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Frame-rate speed, Savitzky-Golay first derivative over 0.75 s."""
    n = len(pos)
    w = min(9, n - 1)
    if w % 2 == 0:
        w -= 1
    dt = float(np.median(np.diff(t_frame)))
    vel = savgol_filter(pos, w, 3, deriv=1, delta=dt, axis=0, mode="interp")
    return np.linalg.norm(vel, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3a_v1")
    ap.add_argument("--video-root", default="data/raw/tii_video")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--calib-dir", default="data/raw/tii_video/calibration")
    ap.add_argument("--out", default="data/processed/p3a")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--clip-len", type=float, default=CLIP_LEN_S)
    args = ap.parse_args()

    out = ROOT / args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []

    for flight in FLIGHTS:
        control, track = flight_parts(flight)
        clip = load_tii_video(ROOT / args.video_root, flight,
                              mocap_dir=ROOT / args.mocap_root,
                              calib_dir=ROOT / args.calib_dir, stride=args.stride)
        # the zips live on the Modal Volume, not on this disk, so the frame names come
        # from the CSV table (same order, same stride as the Phase 1 GeoCalib run)
        cam = frame_table(ROOT / args.video_root, flight)
        names = list(cam["filename"].to_numpy()[:: args.stride])
        if len(names) != len(clip.t):
            raise SystemExit(f"{flight}: {len(names)} frame names vs {len(clip.t)} poses")
        R_bw = Rotation.from_quat(clip.quat).as_matrix()
        R_cb = camera_to_body_pitched(np.deg2rad(clip.cam_uptilt_deg))
        R_cw = np.einsum("nij,jk->nik", R_bw, R_cb)
        imp, speed_mocap = implied_axis_at(
            clip.t, ROOT / args.mocap_root / control / f"mocap_{flight}.csv")
        speed = frame_speed(clip.t, clip.pos)

        # per-frame rotations for each condition
        maps: dict[str, np.ndarray] = {}
        angles: dict[str, np.ndarray] = {}
        for cond in CONDITIONS:
            if cond == "tilt_k0":
                M = condition_maps("tilt_k", R_bw, R_cw, k=0.0)
            elif cond == "tilt_k05":
                M = condition_maps("tilt_k", R_bw, R_cw, k=0.5)
            elif cond == "tilt_k15":
                M = condition_maps("tilt_k", R_bw, R_cw, k=1.5)
            elif cond == "mirror_roll":
                M = condition_maps("mirror_roll", R_bw, R_cw)
            elif cond == "identity":
                M = condition_maps("identity", R_bw, R_cw)
            elif cond == "yaw_control":
                # magnitude matched frame-by-frame to the largest edit (tilt removal)
                M_k0 = condition_maps("tilt_k", R_bw, R_cw, k=0.0)
                mag = np.minimum(rotation_angle_deg(M_k0), 60.0)
                M = condition_maps("yaw_control", R_bw, R_cw, yaw_rot_rad=np.deg2rad(mag))
            else:
                raise ValueError(cond)
            maps[cond] = M.astype(np.float32)
            angles[cond] = rotation_angle_deg(M)

        # clip table (non-overlapping, t measured from the first frame)
        t0 = clip.t[0]
        n_clips = int(np.floor((clip.t[-1] - t0) / args.clip_len))
        for i in range(n_clips):
            lo, hi = t0 + i * args.clip_len, t0 + (i + 1) * args.clip_len
            idx = np.where((clip.t >= lo) & (clip.t < hi))[0]
            if len(idx) < 24:
                continue
            clips.append({
                "clip_id": f"{flight}#{i:02d}", "flight": flight, "control": control,
                "track": track, "index": [int(idx[0]), int(idx[-1]) + 1],
                "t_start": float(clip.t[idx[0]]), "t_end": float(clip.t[idx[-1]]),
                "n_frames": int(len(idx)),
                "speed_median": float(np.median(speed[idx])),
                "speed_mocap_median": float(np.median(speed_mocap[idx])),
                "tilt_implied_median": float(np.nanmedian(tilt_deg(imp[idx]))),
                "uptilt_deg": clip.cam_uptilt_deg,
                "edit_magnitude_deg": {c: float(np.median(angles[c][idx])) for c in CONDITIONS},
            })

        payload = {
            "t": clip.t, "names": np.array(names), "pos": clip.pos, "quat": clip.quat,
            "implied_axis": imp, "speed": speed, "speed_mocap": speed_mocap,
            "tilt_implied_deg": tilt_deg(imp),
            "body_tilt_deg": tilt_deg(np.einsum("nij,j->ni", R_bw, np.array([0.0, 0.0, 1.0]))),
            "uptilt_deg": clip.cam_uptilt_deg, "fps": clip.fps,
            "measured_fps": clip.meta["measured_fps"],
            "image_size": clip.intrinsics["image_size"],
        }
        for c in CONDITIONS:
            payload[f"M_{c}"] = maps[c]
            payload[f"angle_{c}"] = angles[c].astype(np.float32)
        np.savez_compressed(out / f"{flight}.npz", **payload)
        print(f"{flight}: {len(names)} frames, conditions "
              + ", ".join(f"{c} {np.median(angles[c]):.1f}deg" for c in CONDITIONS), flush=True)

    (out / "clips.json").write_text(json.dumps(
        {"config": {"stride": args.stride, "clip_len_s": args.clip_len,
                    "sg_window_s": SG_WINDOW_S, "k_d": K_D, "conditions": CONDITIONS,
                    "implied": "mocap position, native rate, sampled at frame times"},
         "clips": clips}, indent=1))
    print(f"{len(clips)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
