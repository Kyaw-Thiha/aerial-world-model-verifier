"""Track E Task 1: audit of the UZH-FPV data and truth, before any pixel estimation.

Answers, with evidence rather than assumption:

1. **Ground-truth frame and convention.** Which body frame the released pose refers to, which
   quaternion direction, and whether the world is z-up with gravity along -z.
   Test: the flatness identity must hold in the *stated* frame. The body z axis from the
   quaternion is compared with the thrust axis implied by the path, for the quaternion read
   as body->world and inverted. The correct reading gives a residual of a few degrees; the
   wrong one cannot. The world up direction is then left as a free 2-parameter fit and
   compared with +z.
2. **Mount.** Per rig, from the published Kalibr ``camchain-imucam``: optical-axis elevation
   in the body frame, compared with the rig's name (forward vs 45 down). This is a check on
   the extrinsic's *interpretation*, not on the extrinsic itself.
3. **Time offset.** From ``remote/e_sync.py``: the IMU-vs-truth clock offset measured by
   correlating |omega|, its correlation peak and its drift, plus the published Kalibr
   camera-IMU timeshift.
4. **Rig/environment inventory and truth coverage**, and the published vs measured speed.

    pixi run python experiments/e_uzh_audit.py --root data/uzh --out results/e_audit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from src.depicted.metrics import summarize
from src.depicted.uzh import (SEQUENCE_ENV, SEQUENCE_RIG, all_sequences, dense_kinematics,
                              load_sync, load_truth, optical_axis_elevation_deg,
                              published_mount, required_tilt_deg)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def up_direction_fit(R_bw: np.ndarray, z_body: np.ndarray, a: np.ndarray, v: np.ndarray,
                     k_d: float = 0.0, g: float = 9.81) -> dict:
    """Fit the world-frame up direction that makes flatness consistent.

    For a candidate up direction ``n`` the implied thrust axis is
    ``u = a + g n + k_d v`` normalised (``n`` is *down*, i.e. gravity along +g n). The loss is
    the median angle between ``u`` and the measured body z axis. Two free parameters (the
    direction). If the released world frame is z-up the fitted ``n`` comes back as -z.
    """
    ok = np.isfinite(a).all(axis=1) & (np.linalg.norm(a, axis=1) > 0)
    a, v, zb = a[ok], v[ok], z_body[ok]

    def loss(theta):
        n = np.array([np.sin(theta[0]) * np.cos(theta[1]),
                      np.sin(theta[0]) * np.sin(theta[1]),
                      np.cos(theta[0])])
        u = a + g * n + k_d * v
        nu = np.linalg.norm(u, axis=1)
        good = nu > 1e-6
        if good.sum() < 10:
            return 180.0
        u = u[good] / nu[good, None]
        c = np.clip(np.einsum("ij,ij->i", u, zb[good]), -1.0, 1.0)
        return float(np.median(np.degrees(np.arccos(c))))

    best = None
    for th0 in (np.pi, 0.0):  # start at -z (down) and +z
        r = minimize(loss, x0=[th0, 0.0], method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 600})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    n = np.array([np.sin(th[0]) * np.cos(th[1]), np.sin(th[0]) * np.sin(th[1]), np.cos(th[0])])
    return {"up_world": [-float(x) for x in n], "horizon_angle_deg": float(np.degrees(np.arccos(-n[2]))),
            "median_residual_deg": float(best.fun),
            "residual_median": float(best.fun)}


def audit_sequence(root: Path, seq: str, sync: dict, k_d: float = 0.0,
                   window_s: float = 0.10) -> dict:
    truth = load_truth(root, seq)
    gt = np.load(root / seq / "truth.npz")["gt"]
    kin = dense_kinematics(gt, window_s=window_s)
    t_gt = kin["t"] - (truth.t_unix[0] + truth.offset_s)
    # sample the dense derivatives at the frame times
    a = np.column_stack([np.interp(truth.t, t_gt, kin["a"][:, k]) for k in range(3)])
    v = np.column_stack([np.interp(truth.t, t_gt, kin["v"][:, k]) for k in range(3)])
    speed = np.linalg.norm(v, axis=1)
    z_body = truth.R_bw[:, :, 2]
    implied_tilt, u_imp, ok = required_tilt_deg(a, v, k_d=k_d)

    # --- quaternion direction: body->world as published, versus inverted
    res_fwd = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u_imp, z_body), -1, 1)))
    R_inv = np.transpose(truth.R_bw, (0, 2, 1))
    z_inv = R_inv[:, :, 2]
    res_inv = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u_imp, z_inv), -1, 1)))

    fit = up_direction_fit(truth.R_bw, z_body, a, v, k_d=k_d)
    # a drag term can absorb a tilted "gravity", so the same fit is repeated with the TII
    # drag value: if the fitted tilt shrinks, the tilt was drag, not levelling error
    fit_drag = up_direction_fit(truth.R_bw, z_body, a, v, k_d=0.295)
    res_drag = np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", (a + 9.81 * np.array([0.0, 0.0, 1.0]) + 0.295 * v)
        / np.maximum(np.linalg.norm(a + 9.81 * np.array([0.0, 0.0, 1.0]) + 0.295 * v,
                                   axis=1, keepdims=True), 1e-9), z_body), -1, 1)))
    rest = {"n_frames": int(min(5, len(truth.t))),
            "tilt_deg_median": float(np.median(truth.tilt_deg[:5])),
            "accel_norm_median_mps2": float(np.median(np.linalg.norm(a[:5], axis=1)))}

    R_cb, t_cam_imu = published_mount(truth.meta)
    rot_err = float(np.max(np.abs(np.einsum("nij,nkj->nik", truth.R_bw, truth.R_bw)
                                  - np.eye(3))))
    rec = {
        "seq": seq, "rig": truth.rig, "environment": truth.env,
        "n_frames_used": truth.n_frames,
        "n_frames_in_recording": int(truth.meta["n_frames_inside_truth"]) if "n_frames_inside_truth" in truth.meta else None,
        "fps_measured": float(truth.meta["fps_measured"]),
        "truth_span_s": float(gt[-1, 0] - gt[0, 0]),
        "truth_rows": int(len(gt)),
        "truth_dt_s": float(np.median(np.diff(gt[:, 0]))),
        "clock": {k: sync.get(k) for k in ("offset_gt_minus_imu_s", "offset_peak_corr",
                                           "offset_minus_epoch_diff_s", "drift_span_s",
                                           "images_minus_imu_first_s")} if sync else None,
        "clock_offset_applied_s": truth.offset_s,
        "kalibr_timeshift_cam_imu_s": float(truth.meta["kalibr_imu_cam0"]["timeshift_cam_imu"]),
        "quat_orthonormality_max_err": rot_err,
        "quat_convention": {
            "body_to_world_median_residual_deg": float(np.nanmedian(res_fwd)),
            "world_to_body_median_residual_deg": float(np.nanmedian(res_inv)),
            "reading_used": "body_to_world" if np.nanmedian(res_fwd) < np.nanmedian(res_inv)
                            else "world_to_body",
        },
        "world_frame": fit,
        "world_frame_with_drag": fit_drag,
        "flatness_residual_kd0295": summarize(res_drag),
        "rest_segment": rest,
        "mount": {
            "R_cb": R_cb.tolist(), "t_cam_imu_m": t_cam_imu.tolist(),
            "optical_axis_elevation_deg": optical_axis_elevation_deg(R_cb),
        },
        "kinematics": {
            "speed_median_mps": float(np.median(speed)),
            "speed_p95_mps": float(np.percentile(speed, 95)),
            "speed_max_mps": float(speed.max()),
            "implied_tilt_median_deg": float(np.nanmedian(implied_tilt[ok])),
            "implied_tilt_p90_deg": float(np.nanpercentile(implied_tilt[ok], 90)),
            "frac_implied_tilt_ge_10deg": float(np.nanmean(implied_tilt[ok] >= 10.0)),
            "body_tilt_median_deg": float(np.median(truth.tilt_deg)),
        },
        "flatness_residual_kd0": summarize(res_fwd),
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uzh", help="prepared sequences (from Modal)")
    ap.add_argument("--out", default="results/e_audit")
    ap.add_argument("--k-d", type=float, default=0.0,
                    help="drag used in the convention test; 0 keeps the test assumption-free")
    ap.add_argument("--window-s", type=float, default=0.10)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sync = load_sync(root)

    recs = []
    for seq in all_sequences():
        if not (root / seq / "truth.npz").exists():
            print(f"skip {seq}: not prepared")
            continue
        rec = audit_sequence(root, seq, sync.get(seq, {}), k_d=args.k_d,
                             window_s=args.window_s)
        recs.append(rec)
        q = rec["quat_convention"]
        print(f"{seq:20s} {rec['rig']:16s} n={rec['n_frames_used']:4d} "
              f"quat b2w={q['body_to_world_median_residual_deg']:6.2f} "
              f"inv={q['world_to_body_median_residual_deg']:6.2f} "
              f"up={np.round(rec['world_frame']['up_world'], 4).tolist()} "
              f"mount_elev={rec['mount']['optical_axis_elevation_deg']:5.1f} "
              f"tilt_med={rec['kinematics']['implied_tilt_median_deg']:5.1f} "
              f"v95={rec['kinematics']['speed_p95_mps']:5.1f}", flush=True)

    (out / "audit.json").write_text(json.dumps(recs, indent=1))
    print(f"\nwrote {out / 'audit.json'} ({len(recs)} sequences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
