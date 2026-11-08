"""Phase 1b: depicted attitude over time from VO rotation fused with gravity.

Consumes one DPVO run (``results``/Volume ``dpvo/<tag>/poses.npz``) and one
GeoCalib run (``geocalib/<tag>/<flight>.jsonl``), then

1. converts the VO relative rotations into a per-frame camera attitude and
   gravity-aligns the VO frame with one rotation per clip (spread reported),
2. runs the complementary filter at several gains and reports, per gain, the
   body-z tilt error against mocap and the lean-direction error,
3. picks the gain on the fit flights only (leave-one-flight-out) and reports the
   held-out numbers, so no gate number is tuned on the flight it is scored on.

The VO frame's azimuth is arbitrary, so mocap comparison uses one constant yaw
per clip (evaluation only, recorded in the metrics file).

    pixi run -e gpu python experiments/phase1b_fusion.py --dpvo-tag 01a --gc-tag prior_all
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.datasets.tii_video import flight_parts
from src.fusion import (body_z_from_camera, complementary_filter, gravity_align,
                        lean_error_deg, relative_rotations)
from src.perception import frames as F

GAINS = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def poses_to_rot(poses: np.ndarray, convention: str = "camera_to_world") -> np.ndarray:
    """DPVO pose rows (tx ty tz qx qy qz qw) -> (N, 3, 3) camera<->VO-world.

    The upstream demo feeds these to TUM's ``PoseTrajectory3D`` as camera-to-world, so
    that is the default; ``world_to_camera`` transposes them, which is the obvious
    alternative failure mode and is testable because the rotation *angle* between
    consecutive frames is invariant under inversion.
    """
    q = poses[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    R = Rotation.from_quat(q).as_matrix()
    return R if convention == "camera_to_world" else np.transpose(R, (0, 2, 1))


def tilt_of(z: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(z[:, 2] / np.linalg.norm(z, axis=1), -1, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpvo-root", default="data/raw/dpvo")
    ap.add_argument("--dpvo-tag", required=True)
    ap.add_argument("--geocalib-root", default="data/raw/geocalib")
    ap.add_argument("--gc-tag", required=True)
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--fit-flights", nargs="*", default=None)
    ap.add_argument("--dpvo-glob", default="*_stride10_*/poses.npz",
                    help="which DPVO runs to score; must match the GeoCalib stride")
    ap.add_argument("--pose-convention", default="camera_to_world",
                    choices=["camera_to_world", "world_to_camera"])
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = Path("results/phase1b_fusion") / run_id
    out.mkdir(parents=True, exist_ok=True)
    dpvo_dir = Path(args.dpvo_root) / args.dpvo_tag
    gc_dir = Path(args.geocalib_root) / args.gc_tag
    cfg = {"dpvo_tag": args.dpvo_tag, "geocalib_tag": args.gc_tag, "gains": GAINS,
           "pose_convention": args.pose_convention,
           "speed_bins": SPEED_BINS, "fit_flights": args.fit_flights}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    per_flight: dict = {}
    npx = sorted(dpvo_dir.glob(args.dpvo_glob or "*/poses.npz"))
    if not npx:
        raise SystemExit(f"no DPVO pose files under {dpvo_dir}")
    for npz in npx:
        flight = npz.parent.name.split("_stride")[0]
        jl = gc_dir / f"{flight}.jsonl"
        if not jl.exists():
            print(f"{flight}: no GeoCalib rows, skipped")
            continue
        d = np.load(npz)
        poses, frame_ts = d["poses"], d["frame_ts_us"]
        if len(poses) < 20:
            print(f"{flight}: only {len(poses)} poses, skipped")
            continue
        gc = {}
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                gc[int(Path(r["frame"]).stem.split("_")[1])] = np.asarray(r["gravity_up_cam"])
        keep = [i for i, t in enumerate(frame_ts) if int(t) in gc]
        if len(keep) < 20:
            print(f"{flight}: {len(keep)} frames with both inputs, skipped")
            continue
        keep = np.asarray(keep)
        R_vo = poses_to_rot(poses[keep], args.pose_convention)
        up_cam = np.stack([gc[int(frame_ts[i])] for i in keep])
        up_cam /= np.linalg.norm(up_cam, axis=1, keepdims=True)
        control, track = flight_parts(flight)
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        R_cb = F.camera_to_body_pitched(np.radians(clip.cam_uptilt_deg))
        t_rel = (frame_ts[keep] - clip.meta["epoch_start_us"]) * 1e-6
        idx = np.array([int(np.argmin(np.abs(clip.t - t))) for t in t_rel])
        R_bw_true = F.body_to_world(clip.quat[idx])
        z_true = np.einsum("nij,j->ni", R_bw_true, np.array([0.0, 0.0, 1.0]))
        speed = np.linalg.norm(np.gradient(clip.pos, clip.t, axis=0)[idx], axis=1)

        gal = gravity_align(R_vo, up_cam)
        # run the filter in the gravity-aligned frame, otherwise its correction
        # target (world up) is e3 in a frame where up is not e3
        R_al = np.einsum("ij,njk->nik", gal["R_align"], R_vo)
        rel = relative_rotations(R_al)
        # mount-free VO rotation check: the rotation *angle* between consecutive
        # frames is invariant to the camera-to-body rotation and the VO frame, so it
        # can be compared with mocap without any alignment
        ang_vo = np.degrees(Rotation.from_matrix(rel).magnitude())
        ang_true = np.degrees(Rotation.from_matrix(
            relative_rotations(F.body_to_world(clip.quat[idx]))).magnitude())
        vo_rot_check = {"median_abs_err_deg": float(np.median(np.abs(ang_vo - ang_true))),
                        "median_scale_ratio": float(np.median(ang_vo / np.maximum(ang_true, 1e-9))),
                        "median_true_step_deg": float(np.median(ang_true))}
        rec = {"n": int(len(keep)), "vo_relative_rotation_check": vo_rot_check,
               "gravity_align": {k: (v.tolist() if isinstance(v, np.ndarray)
                                                          else v) for k, v in gal.items()},
               "uptilt_metadata_deg": clip.cam_uptilt_deg,
               "true_tilt_deg_median": float(np.median(tilt_of(z_true))),
               "gains": {}}
        for g in GAINS:
            res = complementary_filter(rel, up_cam, gain=g)
            z_est = body_z_from_camera(res.R, R_cb)
            tilt_est, tilt_tru = tilt_of(z_est), tilt_of(z_true)
            err = tilt_est - tilt_tru
            lean = lean_error_deg(z_est, z_true, min_tilt_deg=10.0)
            # vector angle between the estimated and true thrust axis: a scalar tilt
            # difference cannot see an azimuth error, this can
            cos = ((z_est * z_true).sum(axis=1)
                   / (np.linalg.norm(z_est, axis=1) * np.linalg.norm(z_true, axis=1)))
            axis_vec = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            rec["gains"][str(g)] = {
                "tilt_rmse_deg": float(np.sqrt(np.mean(err**2))),
                "tilt_median_abs_deg": float(np.median(np.abs(err))),
                "lean_median_deg": lean.get("median_abs_deg"),
                "lean_n": lean.get("n", 0),
                "yaw_align_rad": lean.get("yaw_align_rad"),
                "up_err_after_median_deg": float(np.median(res.up_err_after_deg)),
                "axis_vector_median_deg": float(np.median(axis_vec)),
                "axis_vector_rmse_deg": float(np.sqrt(np.mean(axis_vec**2)))}
            if g == GAINS[0]:
                np.savez_compressed(out / f"frames_{flight}.npz", z_est=z_est, z_true=z_true,
                                    speed=speed, tilt_est=tilt_est, tilt_tru=tilt_tru)
        per_flight[flight] = rec
        best = min(GAINS, key=lambda g: per_flight[flight]["gains"][str(g)]["tilt_rmse_deg"])
        print(f"{flight:24s} n={len(keep):4d} vo_rot_err={vo_rot_check['median_abs_err_deg']:5.2f} "
              f"align_spread={gal['spread_median_deg']:5.2f} "
              f"best_gain={best} rmse={per_flight[flight]['gains'][str(best)]['tilt_rmse_deg']:6.2f} "
              f"lean={per_flight[flight]['gains'][str(best)]['lean_median_deg']}", flush=True)

    # leave-one-flight-out gain selection: gain picked on the others, scored on the held-out one
    loo = {}
    names = list(per_flight)
    for held in names:
        fit = [f for f in names if f != held] or names
        g_star = min(GAINS, key=lambda g: np.mean(
            [per_flight[f]["gains"][str(g)]["tilt_rmse_deg"] for f in fit]))
        loo[held] = {"gain": g_star, **per_flight[held]["gains"][str(g_star)]}
    metrics = {"run_id": run_id, "git": git_hash(), "config": cfg, "flights": per_flight,
               "leave_one_flight_out": loo}
    if loo:
        rmses = [v["tilt_rmse_deg"] for v in loo.values()]
        leans = [v["lean_median_deg"] for v in loo.values() if v.get("lean_median_deg")]
        metrics["pooled"] = {
            "median_flight_tilt_rmse_deg": float(np.median(rmses)),
            "gate1_tilt_target_deg": 5.0,
            "gate1_tilt_pass": bool(np.median(rmses) < 5.0),
            "median_flight_lean_deg": float(np.median(leans)) if leans else None,
            "gate1_lean_target_deg": 15.0,
            "gate1_lean_pass": bool(leans and np.median(leans) < 15.0)}
        print("pooled:", json.dumps(metrics["pooled"], indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    for f, rec in per_flight.items():
        ys = [rec["gains"][str(g)]["tilt_rmse_deg"] for g in GAINS]
        ax.semilogx(GAINS, ys, "o-", ms=4, lw=1.1, label=f.replace("flight-", ""))
    ax.axhline(5.0, color="k", ls="--", lw=1, label="Gate 1 target (5 deg)")
    ax.set_xlabel("complementary-filter gain per frame")
    ax.set_ylabel("tilt RMSE vs mocap (deg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "gain_sweep.png", dpi=150)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "git.txt").write_text(git_hash() + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
