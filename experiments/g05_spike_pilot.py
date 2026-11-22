"""G05 - pipeline check of the path-demand machinery on Track I's three simulated clips.

EXPLORATORY. Track I has no I02 labelled set, only the I01 spike: one flight rendered three
ways (feasible / level camera / mirrored roll), 91 frames each, full-clip violations. Three
clips cannot calibrate a threshold, so this script reports raw window statistics and does not
choose or tune anything. It exists to answer one question before the real set arrives: does the
path-demand classification and the residual statistic behave on renders whose truth is exact?

Reads Track I's worktree read-only (``--i-dir``), whose per-frame rows already carry
``lean_est`` (reader), ``lean_req`` (path) and ``lean_gt`` (truth), and ``flight_gt.npz`` whose
``yaw``/``frame_idx`` supply the yaw the residual construction needs.

    PYTHONPATH=. python experiments/g05_spike_pilot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.verdict.segments import frame_residual_deg, window_statistics
from src.verdict.schema import Clip

CLIPS = {"feasible": "i01_feasible_pinhole_prior.json",
         "level_camera": "i01_level_pinhole_prior.json",
         "mirror_roll": "i01_mirror_pinhole_prior.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-dir", default="../awmv-track-i/results/i01_spike")
    ap.add_argument("--out", default="results/g05_spike_pilot")
    args = ap.parse_args()
    i_dir = Path(args.i_dir)
    gt = np.load(i_dir / "flight_gt.npz")
    yaw_all = np.asarray(gt["yaw"], float)
    idx = np.asarray(gt["frame_idx"], int)
    t_all = np.asarray(gt["frame_t"], float)
    yaw = yaw_all[idx]

    out = {"status": "EXPLORATORY pilot; 3 clips; no calibration", "clips": {}}
    for name, fname in CLIPS.items():
        rows = json.loads((i_dir / "geocalib" / fname).read_text())["rows"]
        n = len(rows)
        lean_est = np.array([r["lean_est"] for r in rows], float)
        lean_req = np.array([r["lean_req"] for r in rows], float)
        lean_gt = np.array([r["lean_gt"] for r in rows], float)
        tilt_req = np.degrees(np.arcsin(np.clip(np.linalg.norm(lean_req, axis=1), 0, 1)))
        clip = Clip(t=np.asarray(t_all[:n], float), pos=np.zeros((n, 3)), yaw=np.asarray(yaw[:n], float),
                    depicted_lean=lean_est, depicted_conf=np.full(n, np.nan), meta={},
                    lean_req=lean_req, vel_req=np.zeros((n, 2)), tilt_req_deg=tilt_req)
        stats = window_statistics(clip, length=8, hop=4)
        resid_gt = frame_residual_deg(lean_gt, lean_req, np.asarray(yaw[:n], float))
        dep_tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(lean_est, axis=1), 0, 1)))
        win = [{"start": s.start, "stop": s.stop, "residual": s.residual,
                "demand": bool(np.sum(tilt_req[s.start:s.stop] >= 15.0) >= 6),
                "quiet": bool(np.all(tilt_req[s.start:s.stop] <= 3.0))}
               for s in stats if not s.abstained]
        dem = [w["residual"] for w in win if w["demand"] and np.isfinite(w["residual"])]
        qui = [float(np.median(dep_tilt[w["start"]:w["stop"]])) for w in win if w["quiet"]]
        rec = {
            "n_frames": n, "n_windows": len(win),
            "n_demand": int(sum(w["demand"] for w in win)),
            "n_quiet": int(sum(w["quiet"] for w in win)),
            "req_tilt_median_deg": float(np.nanmedian(tilt_req)),
            "demand_residual_median_deg": float(np.median(dem)) if dem else None,
            "demand_residual_max_deg": float(np.max(dem)) if dem else None,
            "quiet_dep_tilt_median_deg": float(np.median(qui)) if qui else None,
            "frame_residual_median_deg": float(np.nanmedian(
                frame_residual_deg(lean_est, lean_req, np.asarray(yaw[:n], float)))),
            "truth_residual_median_deg": float(np.nanmedian(resid_gt)),
        }
        out["clips"][name] = rec
        print(f"{name:14s} req {rec['req_tilt_median_deg']:5.1f} deg | demand windows "
              f"{rec['n_demand']:2d} (median resid {rec['demand_residual_median_deg']}) | "
              f"quiet {rec['n_quiet']:2d} (dep tilt {rec['quiet_dep_tilt_median_deg']}) | "
              f"frame resid {rec['frame_residual_median_deg']:5.2f} (truth {rec['truth_residual_median_deg']:.2f})")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
