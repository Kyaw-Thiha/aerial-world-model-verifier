"""Track C, Task 0: reproduce Track A's DPVO translation number from stored poses.

Read-only on Track A's artifacts. For each stored ``poses.npz`` (DPVO, camera-to-world,
rows ``tx ty tz qx qy qz qw``, stride 10 of the 120 Hz camera = ~12 Hz) it compares the
VO camera path to mocap position at the same timestamps under four alignments:

* ``track_a_scalar``  : Track A's Phase 2 "oracle" (a scalar scale + translation, NO
                        rotation; ``experiments/phase2_implied.py::fit_scale_oracle``)
                        applied to the raw VO frame,
* ``yaw4``            : scale + yaw about z + translation, given the true gravity axis
                        (4-DoF; what a gravity-aware extractor can be aligned with),
* ``sim3``            : Umeyama similarity (7-DoF),
* ``se3``             : Umeyama rigid, scale fixed to 1 (metric error; DPVO has no scale).

The point is to separate "the path is unrecoverable" from "the path was compared without a
rotation alignment". Nothing here is tuned; it only reads stored files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.fusion import attitude as F


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares R, s, t with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, s, t


def yaw_scale_fit(src: np.ndarray, dst: np.ndarray):
    """4-DoF fit: dst ~ s Rz(yaw) src + t, assuming both frames are already z-up."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    c = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()
    d = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    yaw = np.arctan2(d, c)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    s = float(((b @ R) * a).sum() / (a * a).sum())
    t = mu_d - s * R @ mu_s
    return R, s, t


def scalar_fit(src: np.ndarray, dst: np.ndarray):
    """Track A's Phase 2 oracle: scalar scale and translation, identity rotation."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    s = float((a * b).sum() / (a * a).sum())
    return np.eye(3), s, dst.mean(0) - s * src.mean(0)


def score(name, fit, p_vo, p_true):
    R, s, t = fit(p_vo, p_true)
    p_al = s * p_vo @ R.T + t
    err = np.linalg.norm(p_al - p_true, axis=1)
    corr = [float(np.corrcoef(p_al[:, i], p_true[:, i])[0, 1]) for i in range(3)]
    return {"alignment": name, "scale": s, "path_len_m": path_length(p_al),
            "ate_rmse_m": float(np.sqrt((err ** 2).mean())), "axis_corr": corr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpvo-root", default="data/raw/dpvo")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--geocalib-root", default="data/raw/geocalib")
    ap.add_argument("--gc-tag", default="div10")
    ap.add_argument("--tag", default="stride10_fov100_s0.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for npz in sorted(Path(args.dpvo_root).glob(f"*_{args.tag}/poses.npz")):
        flight = npz.parent.name.split("_stride")[0]
        control = "piloted" if flight.split("-")[1].endswith("p") else "autonomous"
        d = np.load(npz)
        poses, ts = d["poses"], d["frame_ts_us"]
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        t = (ts - clip.meta["epoch_start_us"]) * 1e-6
        ok = (t >= clip.t[0]) & (t <= clip.t[-1])
        p_true = np.stack([np.interp(t[ok], clip.t, clip.pos[:, i]) for i in range(3)], axis=1)
        p_vo = poses[ok, :3].astype(np.float64)
        rec = {"flight": flight, "n": int(ok.sum()), "duration_s": float(t[ok][-1] - t[ok][0]),
               "mocap_path_len_m": path_length(p_true), "vo_path_len_raw_units": path_length(p_vo),
               "fits": []}
        # the yaw4 fit needs both frames z-up; VO's frame is arbitrary, so use the oracle
        # gravity route: first a rotation-only fit of the VO frame to mocap (Umeyama sim3 R),
        # which gives "VO -> mocap-aligned axes", then report yaw4 on the residual rotation.
        R0, _, _ = umeyama(p_vo, p_true, with_scale=True)
        p_up = p_vo @ R0.T
        rec["fits"].append(score("track_a_scalar", scalar_fit, p_vo, p_true))
        rec["fits"].append(score("sim3", lambda a, b: umeyama(a, b, True), p_vo, p_true))
        rec["fits"].append(score("se3", lambda a, b: umeyama(a, b, False), p_vo, p_true))
        rec["fits"].append(score("yaw4_after_oracle_up", yaw_scale_fit, p_up, p_true))

        # exact Phase 2 route: joint frames with GeoCalib, gravity align, scalar fit
        jl = Path(args.geocalib_root) / args.gc_tag / f"{flight}.jsonl"
        if jl.exists():
            gc = {}
            for line in jl.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    gc[int(Path(r["frame"]).stem.split("_")[1])] = r["gravity_up_cam"]
            keep = np.array([i for i, t in enumerate(ts) if int(t) in gc])
            if len(keep) >= 40:
                R_vo = Rotation.from_quat(
                    poses[keep, 3:7] / np.linalg.norm(poses[keep, 3:7], axis=1, keepdims=True)
                ).as_matrix()
                p_j = poses[keep, :3].astype(np.float64)
                up = np.stack([gc[int(ts[i])] for i in keep])
                up /= np.linalg.norm(up, axis=1, keepdims=True)
                tj = (ts[keep] - clip.meta["epoch_start_us"]) * 1e-6
                p_tj = np.stack([np.interp(tj, clip.t, clip.pos[:, i]) for i in range(3)], axis=1)
                gal = F.gravity_align(R_vo, up)
                p_al = p_j @ gal["R_align"].T
                a_, b_ = p_al - p_al.mean(0), p_tj - p_tj.mean(0)
                s_ph2 = float((a_ * b_).sum() / (a_ * a_).sum())
                p_sc = s_ph2 * p_al
                err = np.linalg.norm(p_sc - p_sc.mean(0) - (p_tj - p_tj.mean(0)), axis=1)
                rec["phase2_exact"] = {
                    "n_joint": int(len(keep)),
                    "dt_s": float(np.median(np.diff(tj))),
                    "scale": s_ph2,
                    "vo_path_after_scalar_m": path_length(p_sc),
                    "vo_path_gravity_aligned_units": path_length(p_al),
                    "ate_rmse_m": float(np.sqrt((err ** 2).mean())),
                    "axis_corr": [float(np.corrcoef(p_sc[:, i], p_tj[:, i])[0, 1]) for i in range(3)],
                }
        rows.append(rec)
        print(f"{flight}: n={rec['n']} mocap {rec['mocap_path_len_m']:.0f} m, vo raw {rec['vo_path_len_raw_units']:.2f} u")
        for f in rec["fits"]:
            print(f"   {f['alignment']:22s} scale {f['scale']:9.4f}  path {f['path_len_m']:7.1f} m  "
                  f"ATE {f['ate_rmse_m']:7.2f} m  corr {np.round(f['axis_corr'], 2)}")
        pe = rec.get("phase2_exact")
        if pe:
            print(f"   phase2_exact: n={pe['n_joint']} dt={pe['dt_s']:.3f}s scale {pe['scale']:8.5f} "
                  f"vo_path {pe['vo_path_after_scalar_m']:6.2f} m (raw {pe['vo_path_gravity_aligned_units']:7.1f} u) "
                  f"ATE {pe['ate_rmse_m']:6.2f} m corr {np.round(pe['axis_corr'], 2)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
