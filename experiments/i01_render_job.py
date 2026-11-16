"""Build the render job for a simulator candidate (Track I spike).

The flight and the scene are renderer-independent; this writes one job per (candidate, clip)
so every adapter draws the same thing, plus the ground truth the scoring step needs:

    <tag>_<clip>_poses.npz    poses (M,4,4) camera-to-world, OpenCV RDF; yaw, tilt_gt_deg,
                              lean_gt (2), lean_req (2), R_cb, uptilt_deg, gt_hz, k_d
    <tag>_<clip>_job.json     spec (npz path, intrinsics, scene layout, samples)

Clips: ``feasible`` (physics attitude) and ``level`` (the same path with the camera held
level -- the tilt-removal defect). ``lean_req`` is what the path demands; ``lean_gt`` is what
the rendered camera actually does. They agree on the feasible clip and disagree on the level
clip by construction, which is the whole test.

    python experiments/i01_render_job.py --backend isaac --out results/i01_spike/render
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.perception.frames import camera_to_body_pitched
from src.sim.scene import layout
from src.verdict.lean import depicted_lean_from_gravity, roll_pitch_to_lean

WIDTH, HEIGHT, HFOV = 640, 360, 90.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["blender", "isaac", "flightmare"])
    ap.add_argument("--out", default="results/i01_spike/render")
    ap.add_argument("--spike", default="results/i01_spike",
                    help="directory holding flight_gt.npz and frames_{feasible,level}.npy")
    ap.add_argument("--tag", default="i01")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--rt-subframes", type=int, default=4)
    args = ap.parse_args()

    spike = Path(args.spike)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = layout(args.seed)
    gt = np.load(spike / "flight_gt.npz")
    idx = gt["frame_idx"]
    yaw = gt["yaw"][idx]
    tilt_req = np.degrees(np.arccos(np.clip(gt["thrust_axis"][idx] @ np.array([0.0, 0.0, 1.0]), -1, 1)))
    lean_req = roll_pitch_to_lean(gt["roll"][idx], gt["pitch"][idx])
    uptilt = float(json.loads((spike / "flight_summary.json").read_text())["camera_uptilt_deg"])
    R_cb = camera_to_body_pitched(np.radians(uptilt))

    for clip in ("feasible", "level", "mirror"):
        poses = np.load(spike / f"frames_{clip}.npy")
        up_cam = np.einsum("nji,j->ni", poses[:, :3, :3], np.array([0.0, 0.0, 1.0]))
        lean_gt = depicted_lean_from_gravity(up_cam, R_cb)
        up_body = up_cam @ R_cb.T
        tilt_gt = np.degrees(np.arccos(np.clip(up_body @ np.array([0.0, 0.0, 1.0]), -1, 1)))
        poses_path = out / f"{args.tag}_{clip}_poses.npz"
        np.savez_compressed(poses_path, poses=poses, yaw=yaw, tilt_gt_deg=tilt_gt,
                            lean_gt=lean_gt, lean_req=lean_req, R_cb=R_cb,
                            uptilt_deg=np.array([uptilt]),
                            gt_hz=gt["gt_hz"], k_d=gt["k_d"],
                            frame_t=gt["frame_t"])
        job = {
            "backend": args.backend,
            "tag": f"{args.tag}_{clip}",
            "clip": clip,
            "flight": "slalom_8ms_amp2_lambda30",
            "poses": str(poses_path.resolve()),
            "out_dir": str((out / args.tag / clip).resolve()),
            "result_json": str((out / args.tag / clip / "render_timing.json").resolve()),
            "width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV,
            "scene_seed": args.seed,
            "scene": lay,
            "samples": args.samples,
            "rt_subframes": args.rt_subframes,
            "engine": "CYCLES", "device": "GPU", "exposure": -3.0, "denoise": True,
            "mount_uptilt_deg": uptilt,
            "mount_R_cb": R_cb.tolist(),
            "gt": {"gt_hz": float(gt["gt_hz"][0]), "k_d": float(gt["k_d"][0]),
                   "n_gt": int(len(gt["t"])), "video_fps": 30.0},
        }
        p = out / f"{args.tag}_{clip}_job.json"
        p.write_text(json.dumps(job, indent=2))
        print(p, "n_frames", len(poses), "tilt_req_med",
              round(float(np.median(tilt_req)), 2), "tilt_gt_med", round(float(np.median(tilt_gt)), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
