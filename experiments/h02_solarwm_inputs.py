"""H02: build SolarWM camera-pose inputs (5 frames x 5 paths x 3 arms + probes + seeds).

SolarWM consumes an image, a text prompt and an absolute OpenCV camera-to-world trajectory
[frames,4,4] with per-frame intrinsics; its Wan backend rebases the trajectory to the first
frame, so only frame-to-frame attitude changes are controllable. Poses are built with the
verified per-step flatness construction from the pose-injection track (intrinsic ZYX,
camera mount camera_to_body_pitched(0)).

    pixi run python experiments/h02_solarwm_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from src.generation.attitude import body_rotation, body_z_from_poses, roll_pitch_from_thrust
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import G0, implied_thrust_axis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/h02_solarwm_inputs"
FRAMES = ROOT / "data/raw/anwm_frames"
SPECS = ROOT / "results/anwm_clips/specs.json"

N_OUT = 81
DT_S = 1.0
K_D = 0.295
PAD_W, PAD_H = 864, 480
F_PX = 240.0
MAX_TRANSLATION_M = 16.0   # SolarWM's camera guards reject relative translations > 20 m

PATHS = {
    "cruise": "C_cruise_ctl_course_0",
    "turn5": "C_turn_5_fixed_0",
    "turn10": "C_turn_10_fixed_0",
    "turn20": "C_turn_20_fixed_0",
    "slalom10": "C_slalom_10_fixed_0",
}
ARMS = ("level", "flatness", "mirrored_roll")
SEED_PATHS = [
    "C_cruise_ctl_course_0", "C_accel_brake_10_course_0", "C_accel_brake_20_course_1",
    "C_climb_turn_10_fixed_0", "C_climb_turn_20_fixed_0", "C_slalom_5_course_0",
    "C_slalom_10_fixed_0", "C_slalom_10_fixed_1", "C_turn_5_fixed_0",
    "C_turn_10_fixed_0",
]
PROBE_PROFILES = {
    "level": (0.0, np.zeros(N_OUT)),
    "p20": (20.0, np.linspace(0.0, 1.0, N_OUT)),
    "m20": (-20.0, np.linspace(0.0, 1.0, N_OUT)),
    "fast20": (20.0, 1.0 - np.abs(2.0 * np.linspace(0.0, 1.0, N_OUT) - 1.0)),
    "p60": (60.0, np.linspace(0.0, 1.0, N_OUT)),
}


def first_frames(n: int = 5) -> list[dict]:
    out = []
    for d in sorted(p for p in FRAMES.iterdir() if p.is_dir()):
        jpgs = sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))
        if jpgs:
            out.append({"traj": d.name, "frame": int(jpgs[0].stem), "path": jpgs[0]})
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} trajectories")
    return out


def padded_frame(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB").resize((PAD_H, PAD_H), Image.BICUBIC)
    a = np.asarray(im)
    left = (PAD_W - PAD_H) // 2
    right = PAD_W - PAD_H - left
    pad = np.pad(a, ((0, 0), (left, right), (0, 0)), mode="edge")
    Image.fromarray(pad).save(dst)


def node_states(positions: np.ndarray) -> dict:
    p = np.asarray(positions, float)
    v = np.diff(p, axis=0) / DT_S
    a = np.zeros_like(p)
    a[1:len(p) - 1] = (v[1:] - v[:-1]) / DT_S
    vm = np.zeros_like(p)
    vm[1:len(p) - 1] = 0.5 * (v[:-1] + v[1:])
    yaw = np.unwrap(np.concatenate([np.arctan2(v[:1, 1], v[:1, 0]),
                                    np.arctan2(v[:, 1], v[:, 0])]))
    return {"pos": p, "vel": vm, "acc": a, "yaw": yaw}


def resample_positions(nodes: dict) -> dict:
    t = np.arange(len(nodes["pos"])) * DT_S
    cs = CubicSpline(t, nodes["pos"], bc_type="natural", axis=0)
    tt = np.linspace(t[0], t[-1], N_OUT)
    pos = cs(tt)
    pos = pos - pos[0]
    m = float(np.linalg.norm(pos, axis=1).max())
    if m > MAX_TRANSLATION_M:
        pos *= MAX_TRANSLATION_M / m
    yaw = np.interp(tt, t, nodes["yaw"])
    return {"pos": pos, "yaw": yaw}


def _to_out(a: np.ndarray) -> np.ndarray:
    return np.interp(np.linspace(0, len(a) - 1, N_OUT), np.arange(len(a)), a)


def build(nodes: dict, arm: str) -> np.ndarray:
    smp = resample_positions(nodes)
    if arm == "level":
        phi = np.zeros(N_OUT)
        theta = np.zeros(N_OUT)
    else:
        phi_n, theta_n = roll_pitch_from_thrust(nodes["z_flat"], nodes["yaw"])
        if arm == "mirrored_roll":
            phi_n = -phi_n
        # A constant absolute attitude is invisible after the model's first-frame
        # rebasing. Bank in from level over the first 12 of 81 frames, then hold the
        # path-required attitude; this is the physically meaningful entry transient.
        ramp = np.clip((np.arange(N_OUT) - 1.0) / 12.0, 0.0, 1.0)
        phi = _to_out(phi_n) * ramp
        theta = _to_out(theta_n) * ramp
    R_bc = camera_to_body_pitched(0.0)
    poses = np.tile(np.eye(4), (N_OUT, 1, 1))
    poses[:, :3, :3] = body_rotation(smp["yaw"], theta, phi) @ R_bc[None]
    poses[:, :3, 3] = smp["pos"]
    return poses


def path_nodes(spec: dict) -> dict:
    st = node_states(np.asarray(spec["positions"], float))
    z = implied_thrust_axis(st["acc"], st["vel"], K_D, G0)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return {"pos": st["pos"], "yaw": st["yaw"], "z_flat": z,
            "acc": st["acc"], "vel": st["vel"]}


def probe_poses(base: np.ndarray, peak: float, frac: np.ndarray) -> np.ndarray:
    out = base.copy()
    for k, f in enumerate(frac):
        ang = np.deg2rad(peak * f)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                       [np.sin(ang), np.cos(ang), 0.0],
                       [0.0, 0.0, 1.0]])
        out[k, :3, :3] = base[k, :3, :3] @ Rz
    return out


def write_sample(sample_id: str, image: Path, poses: np.ndarray, prompt: str,
                 meta: dict) -> None:
    d = OUT / sample_id
    d.mkdir(parents=True, exist_ok=True)
    Image.open(image).save(d / "image.png")
    np.save(d / "poses.npy", poses.astype(np.float32))
    (d / "prompt.txt").write_text(prompt)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s["id"]: s for s in json.loads(SPECS.read_text())}
    frames = first_frames(5)
    manifest = {"frames": [], "samples": [], "n_out": N_OUT,
                "padded": [PAD_W, PAD_H], "max_translation_m": MAX_TRANSLATION_M}
    for fr in frames:
        dst = OUT / "frames" / f"{fr['traj']}_{fr['frame']:03d}.png"
        dst.parent.mkdir(exist_ok=True)
        padded_frame(fr["path"], dst)
        fr["image"] = dst
        manifest["frames"].append({"traj": fr["traj"], "frame": fr["frame"],
                                   "image": str(dst.relative_to(ROOT))})

    prompt = ("Aerial drone footage over a sunlit city with a river, tall buildings and a "
              "hazy horizon, photorealistic, camera moving over the city.")
    # grid
    for fr in frames:
        for name, cid in PATHS.items():
            nodes = path_nodes(specs[cid])
            for arm in ARMS:
                poses = build(nodes, arm)
                sid = f"grid_{fr['traj']}_{fr['frame']:03d}_{name}_{arm}"
                write_sample(sid, fr["image"], poses, prompt,
                             {"kind": "grid", "traj": fr["traj"], "frame": fr["frame"],
                              "path": name, "arm": arm, "source_clip": cid})
                manifest["samples"].append(sid)
    # convention probes on the first frame
    fr = frames[0]
    base_nodes = path_nodes(specs[PATHS["cruise"]])
    base = build(base_nodes, "level")
    for key, (peak, frac) in PROBE_PROFILES.items():
        poses = probe_poses(base, peak, frac) if key != "level" else base
        sid = f"probe_{key}"
        write_sample(sid, fr["image"], poses, prompt,
                     {"kind": "probe", "traj": fr["traj"], "frame": fr["frame"],
                      "peak_deg": peak, "profile": key})
        manifest["samples"].append(sid)
    # seed variance: 10 paths x 3 seeds on the flatness arm, first frame
    for cid in SEED_PATHS:
        nodes = path_nodes(specs[cid])
        poses = build(nodes, "flatness")
        for seed in range(3):
            sid = f"seed_{cid}_s{seed}"
            write_sample(sid, fr["image"], poses, prompt,
                         {"kind": "seed", "traj": fr["traj"], "frame": fr["frame"],
                          "path": "flatness", "source_clip": cid, "seed": seed})
            manifest["samples"].append(sid)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['samples'])} samples under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
