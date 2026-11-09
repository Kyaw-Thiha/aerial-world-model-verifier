"""Task 4 analysis: does DPVO's translation track on a well-calibrated sequence?

Same shape as the TII Phase 2 check, so the two numbers are comparable: oracle scale from
ground truth (least squares, no rotation), per-axis correlation, path-length ratio, and
the implied-tilt RMSE with `k_d` fixed at 0.295.

If this correlates while TII does not (0.01 / -0.09 / 0.13 and a 41 m path against 479 m),
the TII calibration — a pinhole fit of a 175 deg fisheye, fx/fy = 0.749, invertible only to
56 deg — is the likely cause of the TII failure. If it fails here too, the method is.

    pixi run -e gpu python experiments/phase3a_uzh_dpvo.py --poses <poses.npz> --meta <meta.json>
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
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
K_D = 0.295


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def oracle_scale(vo: np.ndarray, gt: np.ndarray, with_rotation: bool = False) -> dict:
    """Least-squares similarity (scale only, or scale + rotation without reflection)."""
    vo0 = vo - vo.mean(axis=0)
    gt0 = gt - gt.mean(axis=0)
    if with_rotation:
        H = vo0.T @ gt0
        U, S, Vt = np.linalg.svd(H)
        R = (Vt.T @ U.T)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        s = S.sum() / (vo0 ** 2).sum()
    else:
        R = np.eye(3)
        s = float((vo0 * gt0).sum() / (vo0 ** 2).sum())
    return {"scale": float(s), "R": R.tolist()}


def implied_axis(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter
    w = 11 if len(pos) > 15 else max(5, (len(pos) // 3) * 2 + 1)
    if w % 2 == 0:
        w -= 1
    dt = float(np.median(np.diff(t)))
    v = savgol_filter(pos, w, 3, deriv=1, delta=dt, axis=0, mode="interp")
    a = savgol_filter(pos, w, 3, deriv=2, delta=dt, axis=0, mode="interp")
    u = a + np.array([0.0, 0.0, 9.81]) + K_D * v
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.where(n > 0, n, 1.0)


def tilt_deg(axis: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="results/phase3a_verdict")
    args = ap.parse_args()

    z = np.load(args.poses, allow_pickle=False)
    poses = z["poses"]
    ts_ns = z["frame_timestamps_ns"].astype(np.float64)
    meta = json.loads(Path(args.meta).read_text())
    gt = np.asarray(meta["gt_tum"], dtype=np.float64)
    gt_t = gt[:, 0]
    gt_pos = gt[:, 1:4]
    gt_R = Rotation.from_quat(gt[:, 4:8]).as_matrix()

    n = min(len(poses), len(ts_ns))
    vo = poses[:n, :3]
    t = (ts_ns[:n] - ts_ns[0]) * 1e-9
    # ground truth on the frame clock, with the GT start removed
    g = np.column_stack([np.interp(ts_ns[:n], gt_t, gt_pos[:, k]) for k in range(3)])
    g = g - g[0]
    rv = Rotation.from_matrix(gt_R).as_rotvec()
    rv = np.column_stack([np.interp(ts_ns[:n], gt_t, rv[:, k]) for k in range(3)])
    gR = Rotation.from_rotvec(rv).as_matrix()

    fit = oracle_scale(vo, g)
    s = fit["scale"]
    vo_s = vo * s
    corr = [float(np.corrcoef(vo_s[:, k], g[:, k])[0, 1]) for k in range(3)]
    path_vo = float(np.linalg.norm(np.diff(vo_s, axis=0), axis=1).sum())
    path_gt = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())

    imp_vo = implied_axis(vo_s, t)
    imp_gt = implied_axis(g, t)
    tilt_vo, tilt_gt = tilt_deg(imp_vo), tilt_deg(imp_gt)
    err = np.abs(tilt_vo - tilt_gt)
    valid = np.isfinite(err) & (tilt_gt > 10.0)
    rmse = float(np.sqrt(np.nanmean((tilt_vo[valid] - tilt_gt[valid]) ** 2))) if valid.any() else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / f"{run_id}_uzh_dpvo"
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_id": out.name, "git": git_hash(), "sequence": meta.get("sequence"),
        "n_frames": int(n), "oracle_scale": s,
        "per_axis_correlation": corr,
        "path_length_vo_m": path_vo, "path_length_gt_m": path_gt,
        "path_ratio": path_vo / path_gt if path_gt else None,
        "implied_tilt_rmse_deg": rmse, "n_tilt_frames": int(valid.sum()),
        "reference_tii": {"per_axis_correlation": [0.01, -0.09, 0.13],
                          "path_vo_m": 41.0, "path_mocap_m": 479.0,
                          "implied_tilt_rmse_deg": 28.97},
        "verdict": ("calibration_is_the_suspect" if (rmse is not None and rmse < 8.0
                                                     and min(corr) > 0.8)
                    else "method_or_further_work"),
        "note": "oracle scale = least squares without rotation, as in Phase 2",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(g[:, 0], g[:, 1], label="ground truth")
    ax[0].plot(vo_s[:, 0], vo_s[:, 1], label="DPVO (oracle scale)")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{meta.get('sequence')}: top view, path {path_vo:.1f} vs {path_gt:.1f} m")
    tt = t
    ax[1].plot(tt, tilt_gt, label="tilt from truth")
    ax[1].plot(tt, tilt_vo, label="tilt from DPVO (oracle scale)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("implied tilt (deg)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=140)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
