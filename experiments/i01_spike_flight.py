"""I01 spike: build one tiny flight and the render jobs for every simulator candidate.

Produces, in ``results/i01_spike/``:

* ``flight_gt.npz``      -- truth at 250 Hz (t, pos, vel, acc, yaw, thrust axis, R_bw)
* ``flight_summary.json``-- the numbers the report quotes (tilt range, speed, frames)
* ``frames_feasible.npz``-- per-render-frame camera-to-world poses (physical attitude)
* ``frames_level.npz``   -- the same path rendered with a level camera (tilt removal defect)
* ``job_blender_{feasible,level}.json`` -- Blender job specs (renderer adapter input)
* ``labels.json``        -- the defect, its size and its interval, for every clip

The two renders are the point of the spike at this stage: the same flight, once with the
attitude physics requires and once with the attitude the camera would have if tilt were
removed. Any renderer that cannot do both with a *known* pose cannot carry this track.

    pixi run python experiments/i01_spike_flight.py --out results/i01_spike
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.sim.flight import Flight, build, pinhole_K

WIDTH, HEIGHT, HFOV = 640, 360, 90.0
UPTILT_DEG = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/i01_spike")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--uptilt", type=float, default=UPTILT_DEG)
    ap.add_argument("--amplitude", type=float, default=2.0)
    ap.add_argument("--wavelength", type=float, default=30.0)
    ap.add_argument("--maneuver", default="slalom",
                    choices=["slalom", "turn", "ellipse", "lemniscate", "hover"])
    args = ap.parse_args()

    out = Path(args.out)
    (out / "jobs").mkdir(parents=True, exist_ok=True)

    # Default: a slalom. Lateral acceleration changes sign twice per wavelength, so the
    # depicted tilt must follow a sinusoid with both rolls exercised, which is what the
    # verifier's gain and sign statistics are built on. A turn gives one constant tilt and
    # is available with --maneuver turn.
    if args.maneuver == "turn":
        f: Flight = build("turn", speed=args.speed, radius=args.radius,
                          arc_deg=float(np.degrees(args.duration * args.speed / args.radius)))
    elif args.maneuver == "slalom":
        f = build("slalom", speed=args.speed, amplitude=args.amplitude,
                  wavelength=args.wavelength, duration=args.duration, height=12.0)
    elif args.maneuver == "ellipse":
        f = build("ellipse", radius=args.radius, speed=args.speed,
                  duration=args.duration)
    elif args.maneuver == "lemniscate":
        f = build("lemniscate", scale=args.radius, speed=args.speed, duration=args.duration)
    else:
        f = build("hover", duration=args.duration)
    f.video_fps = 30.0
    feasible = f.camera_poses(uptilt_deg=args.uptilt, level_camera=False)
    level = f.camera_poses(uptilt_deg=args.uptilt, level_camera=True)
    mirror = f.camera_poses(uptilt_deg=args.uptilt, mirror_roll=True)

    np.savez_compressed(
        out / "flight_gt.npz",
        t=f.t, pos=f.pos, vel=f.vel, acc=f.acc, yaw=f.yaw,
        thrust_axis=f.thrust_axis, R_bw=f.R_bw, roll=f.roll, pitch=f.pitch,
        frame_t=feasible["frame_t"], frame_idx=feasible["frame_idx"],
        gt_hz=np.array([f.gt_hz]), k_d=np.array([f.k_d]),
    )
    np.save(out / "frames_feasible.npy", feasible["poses"])
    np.save(out / "frames_level.npy", level["poses"])
    np.save(out / "frames_mirror.npy", mirror["poses"])

    K = pinhole_K(WIDTH, HEIGHT, HFOV)
    summary = f.summary() | {
        "width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV,
        "K": K.tolist(),
        "k_d": f.k_d,
        "n_render_frames": int(len(feasible["poses"])),
        "tilt_at_first_frame_deg": float(f.tilt_deg()[0]),
        "tilt_at_last_frame_deg": float(f.tilt_deg()[-1]),
    }
    (out / "flight_summary.json").write_text(json.dumps(summary, indent=2))

    labels = {
        "clip_feasible": {"label": "feasible", "violation": None,
                          "interval_s": None, "magnitude_deg": 0.0},
        "clip_level_camera": {
            "label": "infeasible",
            "violation": "tilt_removed",
            "interval_s": [0.0, float(f.t[-1])],
            "fraction_of_flight": 1.0,
            "magnitude_deg": float(np.median(f.tilt_deg())),
            "note": "camera held level along the same path; the path's required tilt is unchanged",
        },
        "clip_mirror_roll": {
            "label": "infeasible",
            "violation": "mirrored_roll",
            "interval_s": [0.0, float(f.t[-1])],
            "fraction_of_flight": 1.0,
            "magnitude_deg": float(2 * np.median(np.abs(np.degrees(f.roll)))),
            "note": "roll sign flipped, yaw/pitch/position unchanged; lean roll component flips sign",
        },
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    for tag, target in (("feasible", "frames_feasible.npy"), ("level", "frames_level.npy"),
                        ("mirror", "frames_mirror.npy")):
        job = {
            "poses": str((out / target).resolve()),
            "out_dir": str((out / "frames_blender" / tag).resolve()),
            "width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV,
            "scene_seed": 7, "scene": "outdoor",
            "engine": "CYCLES", "device": "GPU", "samples": 24,
            "result_json": str((out / "frames_blender" / tag / "render_timing.json").resolve()),
        }
        (out / "jobs" / f"job_blender_{tag}.json").write_text(json.dumps(job, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
