"""Phase 1a: depicted tilt per frame from GeoCalib, against mocap truth.

Reads one GeoCalib JSONL run (per-frame gravity vectors in camera coordinates,
written by ``remote/a_geocalib.py``), pairs each frame with the mocap attitude at
the frame timestamp, and reports the angle between the two up vectors — a
convention-free metric — plus the body tilt and roll/pitch derived from both.

Every per-flight row is reported twice: raw, and after removing **one constant
rotation per flight** (the rotation taking the mean estimated up to the mean true
up). The offset itself is printed; it is never applied silently.

    pixi run python experiments/phase1_depicted.py --tag prior_all_8
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
from src.datasets.tii_video import camera_to_body, flight_parts, load_calibration
from src.perception import frames as F
from src.perception import undistort as U

SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def frame_ts(name: str) -> int:
    return int(Path(name).stem.split("_")[1])


def truth_for_frame(clip, R_cb, name: str) -> tuple[np.ndarray, float, float]:
    """Mocap up in camera coords, speed and body tilt at one frame timestamp."""
    t_rel = (frame_ts(name) - clip.meta["epoch_start_us"]) * 1e-6
    i = int(np.argmin(np.abs(clip.t - t_rel)))
    R_bw = F.body_to_world(clip.quat[i:i + 1])[0]
    up_b = R_bw.T @ np.array([0.0, 0.0, 1.0])
    up_cam = R_cb.T @ up_b
    vel = np.gradient(clip.pos, clip.t, axis=0)[i]
    return up_cam / np.linalg.norm(up_cam), float(np.linalg.norm(vel)), float(
        np.degrees(np.arccos(np.clip(up_b[2], -1, 1))))


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def const_offset_rotation(est: np.ndarray, tru: np.ndarray) -> np.ndarray:
    """Single rotation mapping the mean estimated up onto the mean true up."""
    e = est.mean(axis=0) / np.linalg.norm(est.mean(axis=0))
    t = tru.mean(axis=0) / np.linalg.norm(tru.mean(axis=0))
    v = np.cross(e, t)
    c = float(e @ t)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * ((1 - c) / s**2)


def summarize(err: np.ndarray, speeds: np.ndarray) -> dict:
    out = {"n": int(len(err)),
           "rmse_deg": float(np.sqrt(np.mean(err**2))) if len(err) else None,
           "median_deg": float(np.median(err)) if len(err) else None,
           "p90_deg": float(np.percentile(err, 90)) if len(err) else None,
           "by_speed": {}}
    for lo, hi in SPEED_BINS:
        m = (speeds >= lo) & (speeds < hi)
        if m.sum() >= 5:
            out["by_speed"][f"{lo}-{hi}"] = {
                "n": int(m.sum()), "median_deg": float(np.median(err[m])),
                "rmse_deg": float(np.sqrt(np.mean(err[m] ** 2)))}
    return out


def plots(out: Path, metrics: dict) -> None:
    """Error vs speed (per flight and pooled) and a per-flight bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for flight, rec in metrics["flights"].items():
        bins = list(rec["raw"]["by_speed"].items())
        if not bins:
            continue
        xs = [0.5 * (float(k.split("-")[0]) + min(float(k.split("-")[1]), 24)) for k, _ in bins]
        ys = [v["median_deg"] for _, v in bins]
        ax.plot(xs, ys, "o-", lw=1.2, ms=4, label=flight.replace("flight-", ""))
    ax.axhline(5.0, color="k", ls="--", lw=1, label="Gate 1a target (5 deg)")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("median depicted-tilt error (deg)")
    ax.set_title("Phase 1a: depicted tilt from GeoCalib vs mocap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "error_vs_speed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(metrics["flights"])
    raw = [metrics["flights"][f]["raw"]["rmse_deg"] for f in names]
    off = [metrics["flights"][f]["with_constant_offset"]["rmse_deg"] for f in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, raw, 0.4, label="raw")
    ax.bar(xs + 0.2, off, 0.4, label="one constant rotation per flight")
    ax.axhline(5.0, color="k", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("flight-", "") for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rmse_per_flight.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="GeoCalib run tag under the Volume")
    ap.add_argument("--geocalib-root", default="data/raw/geocalib")
    ap.add_argument("--video-root", default="data/raw/tii_video")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = Path("results/phase1_depicted") / run_id
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.geocalib_root) / args.tag
    summary_in = json.loads((src / "summary.json").read_text())
    cfg = {"tag": args.tag, "geocalib_summary": summary_in, "speed_bins": SPEED_BINS,
           "offset_policy": "one constant rotation per flight, reported not applied by default",
           "mount": "analytic R_cb at the recorded per-flight uptilt"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    metrics: dict = {"run_id": run_id, "git": git_hash(), "config": cfg, "flights": {}}
    for jl in sorted(src.glob("*.jsonl")):
        flight = jl.stem
        control, track = flight_parts(flight)
        cal = load_calibration(Path(args.video_root) / "calibration", control, track)
        diag = U.model_diagnostics(cal["K"], cal["dist"], (640, 480))
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        R_cb = F.camera_to_body_pitched(np.radians(clip.cam_uptilt_deg))

        rows = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
        est, tru, spd, tilt = [], [], [], []
        for r in rows:
            if r.get("gravity_up_cam") is None:
                continue
            u_t, v, th = truth_for_frame(clip, R_cb, r["frame"])
            e = np.asarray(r["gravity_up_cam"], dtype=np.float64)
            e = e / np.linalg.norm(e)
            est.append(e)
            tru.append(u_t)
            spd.append(v)
            tilt.append(th)
        if len(est) < 5:
            print(f"{flight}: only {len(est)} usable frames, skipped")
            continue
        est, tru = np.array(est), np.array(tru)
        spd, tilt = np.array(spd), np.array(tilt)
        err = angle_deg(est, tru)
        R_off = const_offset_rotation(est, tru)
        err_off = angle_deg(est @ R_off, tru)
        rv = np.degrees(Rotation.from_matrix(R_off).as_rotvec())

        rec = {"control": control, "track": track,
               "uptilt_metadata_deg": clip.cam_uptilt_deg,
               "exposure": clip.meta.get("camera_metadata", {}).get("exposure"),
               "diagnostics": {k: diag[k] for k in ("fx_fy_ratio", "corners_representable",
                                                    "theta_at_peak_deg")},
               "true_tilt_deg": {"median": float(np.median(tilt)),
                                 "p90": float(np.percentile(tilt, 90))},
               "raw": summarize(err, spd),
               "with_constant_offset": summarize(err_off, spd),
               "offset_rotvec_deg": [float(x) for x in rv],
               "offset_angle_deg": float(np.linalg.norm(rv))}
        metrics["flights"][flight] = rec
        np.savez_compressed(out / f"frames_{flight}.npz", est_up=est, true_up=tru,
                            speed=spd, true_tilt=tilt, err_deg=err, err_deg_offset=err_off)
        print(f"{flight:24s} n={len(err):4d} rmse={rec['raw']['rmse_deg']:6.2f} "
              f"med={rec['raw']['median_deg']:6.2f} | offset {rec['offset_angle_deg']:5.2f} deg "
              f"-> rmse {rec['with_constant_offset']['rmse_deg']:6.2f}", flush=True)

    allerr = [metrics["flights"][f]["raw"]["rmse_deg"] for f in metrics["flights"]]
    if allerr:
        metrics["pooled"] = {
            "n_flights": len(allerr),
            "mean_of_flight_rmse_deg": float(np.mean(allerr)),
            "median_flight_rmse_deg": float(np.median(allerr)),
            "mean_of_flight_rmse_with_offset_deg": float(np.mean(
                [metrics["flights"][f]["with_constant_offset"]["rmse_deg"]
                 for f in metrics["flights"]])),
            "gate_1a_target_deg": 5.0,
            "gate_1a_pass": bool(np.median(allerr) < 5.0)}
        print("pooled:", json.dumps(metrics["pooled"], indent=2))

    plots(out, metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "git.txt").write_text(git_hash() + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
