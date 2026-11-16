"""H01 Part 2: build the LingBot-World camera-pose inputs.

Five aerial first frames from the ANWM release (five distinct trajectories, the smallest
recorded frame index each), padded from 512x512 to the 480x832 geometry LingBot assumes,
and five designed manoeuvres (cruise, turn 5/10/20 deg, slalom-10). For each frame x path
there are three arms:

* ``level``          — same translations and heading, camera level;
* ``flatness``       — camera attitude from the verifier flatness expression
                       (z_B ~ a + g e3 + k_d v, k_d = 0.295, dt = 1 s), built through the
                       pose-injection track's verified intrinsic-ZYX construction;
* ``mirrored_roll``  — flatness arm with the roll component negated (sign check).

Plus one convention probe: a pure +20 deg camera roll on a straight translation, which
must visibly rotate the horizon (checked before any tilt result is trusted).

Poses are OpenCV camera-to-world (the model's own comment at
``wan/image2video_fast.py``: ``poses.npy # opencv coordinate``), intrinsics are
``[fx, fy, cx, cy]`` for the 480x832 frame the pipeline expects.

    pixi run python experiments/h01_lingbot_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from src.generation.attitude import (body_rotation, body_z_from_poses,
                                     roll_pitch_from_thrust)
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import G0, implied_thrust_axis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/h01_lingbot_inputs"
FRAMES = ROOT / "data/raw/anwm_frames"
SPECS = ROOT / "results/anwm_clips/specs.json"

N_OUT = 49              # frames (4n+1); ~3 s at 16 fps
DT_S = 1.0              # path timing used by the ANWM design
K_D = 0.295
PAD_W, PAD_H = 832, 480
F_PX = 240.0            # 90 deg FOV at 480x480, the ANWM reader's camera
PATHS = {
    "cruise": "C_cruise_ctl_course_0",
    "turn5": "C_turn_5_fixed_0",
    "turn10": "C_turn_10_fixed_0",
    "turn20": "C_turn_20_fixed_0",
    "slalom10": "C_slalom_10_fixed_0",
}
ARMS = ("level", "flatness", "mirrored_roll")


def first_frames(n: int = 5) -> list[dict]:
    out = []
    for d in sorted(p for p in FRAMES.iterdir() if p.is_dir()):
        jpgs = sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))
        if not jpgs:
            continue
        out.append({"traj": d.name, "frame": int(jpgs[0].stem), "path": jpgs[0]})
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} trajectories under {FRAMES}")
    return out


def padded_frame(src: Path, dst: Path) -> np.ndarray:
    im = Image.open(src).convert("RGB").resize((PAD_H, PAD_H), Image.BICUBIC)
    a = np.asarray(im)
    left = (PAD_W - PAD_H) // 2
    right = PAD_W - PAD_H - left
    pad = np.pad(a, ((0, 0), (left, right), (0, 0)), mode="edge")
    Image.fromarray(pad).save(dst)
    return pad


def node_states(positions: np.ndarray) -> dict:
    """Per-step velocity, heading and acceleration at the 17 commanded waypoints.

    This is the construction the ANWM design uses (and the one the pose-injection track
    verified): the whole velocity change between consecutive waypoints happens inside one
    step, so the attitude at waypoint i is the acceleration that produced the frame ending
    there. Ends carry zero acceleration.
    """
    p = np.asarray(positions, float)
    n = len(p)
    v = np.diff(p, axis=0) / DT_S
    a = np.zeros_like(p)
    a[1:n - 1] = (v[1:] - v[:-1]) / DT_S
    vm = np.zeros_like(p)
    vm[1:n - 1] = 0.5 * (v[:-1] + v[1:])
    yaw = np.arctan2(v[:, 1], v[:, 0])
    yaw = np.concatenate([yaw[:1], yaw])
    yaw = np.unwrap(yaw)
    return {"pos": p, "vel": vm, "acc": a, "yaw": yaw}


def resample(pos_nodes: np.ndarray, yaw_nodes: np.ndarray,
             roll_nodes: np.ndarray, pitch_nodes: np.ndarray) -> dict:
    """Smooth positions (cubic spline) and attitudes (SLERP) at N_OUT frame times."""
    t = np.arange(len(pos_nodes)) * DT_S
    cs = CubicSpline(t, pos_nodes, bc_type="natural", axis=0)
    tt = np.linspace(t[0], t[-1], N_OUT)
    pos = cs(tt)
    pos = pos - pos[0]
    # rotations at the nodes, then slerp
    R_nodes = body_rotation(yaw_nodes, pitch_nodes, roll_nodes)
    rot = Slerp(t, Rotation.from_matrix(R_nodes))(tt)
    yaw = np.interp(tt, t, yaw_nodes)
    return {"pos": pos, "yaw": yaw, "rot": rot}


def yaw_from_vel(vel: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    return np.unwrap(yaw)


def mirror_roll(z_B: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Negate the roll component of the thrust axis, keeping heading and pitch."""
    phi, theta = roll_pitch_from_thrust(z_B, yaw)
    R = body_rotation(yaw, theta, -phi)
    return R[:, :, 2]


def build_poses(pos: np.ndarray, yaw: np.ndarray, z_B: np.ndarray) -> np.ndarray:
    R_bc = camera_to_body_pitched(0.0)
    phi, theta = roll_pitch_from_thrust(z_B, yaw)
    R_wb = body_rotation(yaw, theta, phi)
    poses = np.tile(np.eye(4), (len(pos), 1, 1))
    poses[:, :3, :3] = R_wb @ R_bc[None]
    poses[:, :3, 3] = pos
    return poses


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s["id"]: s for s in json.loads(SPECS.read_text())}
    frames = first_frames(5)
    manifest = {"frames": [], "paths": {}, "grid": [], "convention_probe": {},
                "n_out": N_OUT, "dt_s": DT_S, "k_d": K_D, "intrinsics_px": [F_PX, F_PX,
                                                                            PAD_W / 2, PAD_H / 2]}
    (OUT / "frames").mkdir(exist_ok=True)
    for f in frames:
        dst = OUT / "frames" / f"{f['traj']}_{f['frame']:03d}.png"
        padded_frame(f["path"], dst)
        manifest["frames"].append({"traj": f["traj"], "frame": f["frame"],
                                   "file": str(dst.relative_to(ROOT))})

    for name, cid in PATHS.items():
        st = node_states(np.asarray(specs[cid]["positions"], float))
        z_flat = implied_thrust_axis(st["acc"], st["vel"], K_D, G0)
        z_flat = z_flat / np.linalg.norm(z_flat, axis=1, keepdims=True)
        phi, theta = roll_pitch_from_thrust(z_flat, st["yaw"])
        tilt_deg = np.degrees(np.arccos(np.clip(z_flat[:, 2], -1, 1)))
        sampled = {
            "level": resample(st["pos"], st["yaw"], np.zeros_like(phi),
                              np.zeros_like(theta)),
            "flatness": resample(st["pos"], st["yaw"], phi, theta),
            "mirrored_roll": resample(st["pos"], st["yaw"], -phi, theta),
        }
        R_bc = camera_to_body_pitched(0.0)
        variants = {}
        for arm, smp in sampled.items():
            poses = np.tile(np.eye(4), (len(smp["pos"]), 1, 1))
            poses[:, :3, :3] = smp["rot"].as_matrix() @ R_bc[None]
            poses[:, :3, 3] = smp["pos"]
            variants[arm] = poses
        manifest["paths"][name] = {
            "source_clip": cid, "n_waypoints": int(len(specs[cid]["positions"])),
            "step_spacing_m": float(np.mean(np.linalg.norm(
                np.diff(np.asarray(specs[cid]["positions"], float), axis=0), axis=1))),
            "required_tilt_deg_median": float(np.median(tilt_deg)),
            "required_tilt_deg_max": float(np.max(tilt_deg)),
            "roll_deg_min": float(np.degrees(phi).min()),
            "roll_deg_max": float(np.degrees(phi).max()),
        }
        for f in frames:
            for arm in ARMS:
                d = OUT / f"{f['traj']}_{f['frame']:03d}" / name / arm
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "poses.npy", variants[arm])
                np.save(d / "thrust.npy", body_z_from_poses(variants[arm], R_bc))
                K = np.tile(np.array([F_PX, F_PX, PAD_W / 2, PAD_H / 2]), (N_OUT, 1))
                np.save(d / "intrinsics.npy", K)
                manifest["grid"].append({"traj": f["traj"], "frame": f["frame"],
                                         "path": name, "arm": arm,
                                         "image": f"frames/{f['traj']}_{f['frame']:03d}.png",
                                         "dir": str(d.relative_to(ROOT))})

    # convention probe: straight translation, constant +20 deg camera roll
    st = node_states(np.asarray(specs[PATHS["cruise"]]["positions"], float))
    smp = resample(st["pos"], st["yaw"], np.zeros(len(st["yaw"])), np.zeros(len(st["yaw"])))
    base = np.tile(np.eye(4), (len(smp["pos"]), 1, 1))
    base[:, :3, :3] = smp["rot"].as_matrix() @ camera_to_body_pitched(0.0)[None]
    base[:, :3, 3] = smp["pos"]
    roll = np.deg2rad(20.0)
    Rz_cam = np.array([[np.cos(roll), -np.sin(roll), 0.0],
                       [np.sin(roll), np.cos(roll), 0.0],
                       [0.0, 0.0, 1.0]])
    # The model conditions on *relative* frame-to-frame poses (compute_relative_poses,
    # framewise=True), so a constant absolute roll is invisible (every relative rotation
    # is the identity). The probe therefore ramps the roll linearly from 0 to +/-20 deg
    # across the clip, which is the smallest change the model can actually see.
    probes = {}
    ramp = np.linspace(0.0, 1.0, N_OUT)
    for sign in (1.0, -1.0):
        probe = base.copy()
        for k, frac in enumerate(ramp):
            ang = sign * roll * frac
            Rs = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                           [np.sin(ang), np.cos(ang), 0.0],
                           [0.0, 0.0, 1.0]])
            probe[k, :3, :3] = base[k, :3, :3] @ Rs
        d = OUT / "convention_probe" / f"rollramp{int(sign * 20.0):+d}".replace("+", "p")
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "poses.npy", probe)
        np.save(d / "thrust.npy", body_z_from_poses(probe, camera_to_body_pitched(0.0)))
        np.save(d / "intrinsics.npy", np.tile(np.array([F_PX, F_PX, PAD_W / 2, PAD_H / 2]),
                                              (N_OUT, 1)))
        probes[str(int(sign * 20.0))] = {
            "dir": str(d.relative_to(ROOT)), "roll_deg": float(sign * 20.0),
            "roll_profile": "linear ramp 0 -> sign*20 deg",
            "image": f"frames/{frames[0]['traj']}_{frames[0]['frame']:03d}.png"}
    manifest["convention_probe"] = probes
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['grid'])} grid clips + 1 convention probe; "
          f"{len(manifest['frames'])} first frames")
    for name, v in manifest["paths"].items():
        print(name, "required tilt median/max",
              round(v["required_tilt_deg_median"], 2), round(v["required_tilt_deg_max"], 2),
              "roll", round(v["roll_deg_min"], 1), round(v["roll_deg_max"], 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
