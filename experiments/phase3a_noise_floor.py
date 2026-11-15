"""Task 1: GeoCalib's false-alarm floor on ANWM's released (exactly level) frames.

The ANWM release carries roll = pitch = 0 in every frame (checked in
``scripts/inspect_anwm_traj.py`` across 33 test trajectories), and its camera is the
body: a level frame has world up at ``(0, -1, 0)`` in the camera frame, which is also
GeoCalib's own convention for roll = pitch = 0. So the angle between the estimate and
that fixed direction is the estimator's false-alarm floor: any ANWM verdict whose
residual is not clearly above it cannot be trusted.

Input : results/anwm_scoring/<tag>/floor_<size>.jsonl (from remote/a_geocalib_p3.py)
Output: results/anwm_scoring/<run_id>/{noise_floor.json, noise_floor.png}
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

ROOT = Path(__file__).resolve().parents[1]
UP_LEVEL = np.array([0.0, -1.0, 0.0])


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tilt_deg(up: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(up @ UP_LEVEL, -1.0, 1.0)))


def roll_pitch_deg(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed camera roll/pitch implied by an up direction (GeoCalib's own mapping)."""
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    pitch = np.degrees(np.arcsin(np.clip(up[:, 2], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(up[:, 0], -up[:, 1]))
    return roll, pitch


def summarise(up: np.ndarray) -> dict:
    t = tilt_deg(up)
    roll, pitch = roll_pitch_deg(up)
    return {"n": int(len(t)),
            "median_deg": float(np.median(t)),
            "mean_deg": float(np.mean(t)),
            "p90_deg": float(np.percentile(t, 90)),
            "p99_deg": float(np.percentile(t, 99)),
            "max_deg": float(np.max(t)),
            "frac_gt_5deg": float(np.mean(t > 5.0)),
            "frac_gt_10deg": float(np.mean(t > 10.0)),
            "roll_bias_deg": float(np.median(roll)),
            "pitch_bias_deg": float(np.median(pitch))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="anwm_floor_v1")
    ap.add_argument("--dir", default="results/anwm_scoring/anwm_floor_v1")
    ap.add_argument("--out", default="results/anwm_scoring")
    args = ap.parse_args()
    src = ROOT / args.dir
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)

    result = {"source": str(src), "reference_up_cam": UP_LEVEL.tolist(),
              "note": "ANWM's released frames are exactly level (roll = pitch = 0), so "
                      "this is the estimator's false-alarm floor, not a motion estimate",
              "sizes": {}}
    for p in sorted(src.glob("floor_*.jsonl")):
        size = p.stem.replace("floor_", "")
        rows = load_rows(p)
        up = np.array([r["gravity_up_cam"] for r in rows])
        result["sizes"][size] = summarise(up)
        # per-trajectory spread tells us whether a scene dominates the floor
        traj = np.array([r["traj"] for r in rows])
        per = {t: summarise(up[traj == t])["median_deg"] for t in sorted(set(traj))}
        result["sizes"][size]["per_trajectory_median_deg"] = per
        print(f"{size}px: median {result['sizes'][size]['median_deg']:.2f} deg, "
              f"p90 {result['sizes'][size]['p90_deg']:.2f}, "
              f">5 deg {100*result['sizes'][size]['frac_gt_5deg']:.1f}%, "
              f">10 deg {100*result['sizes'][size]['frac_gt_10deg']:.1f}%", flush=True)

    (out / "noise_floor.json").write_text(json.dumps(result, indent=1))
    (out / "git.txt").write_text(git_hash() + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    for size, s in result["sizes"].items():
        p = src / f"floor_{size}.jsonl"
        up = np.array([r["gravity_up_cam"] for r in load_rows(p)])
        t = np.sort(tilt_deg(up))
        ax.plot(t, np.linspace(0, 1, len(t)), label=f"{size} px (median {s['median_deg']:.2f})")
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.axvline(10, color="k", ls="--", lw=1)
    ax.set_xlabel("estimated tilt on level ANWM frames (deg)")
    ax.set_ylabel("empirical CDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "noise_floor.png", dpi=140)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
