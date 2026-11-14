"""Track E Task 3b: check the published camera-IMU mount without any gravity estimator.

E02 left one question open: the 45-degree-down rigs carry a 14.2 deg constant discrepancy
between the depicted gravity and truth, and with a gravity estimator alone it is impossible to
say whether the published `T_cam_imu` is wrong or GeoCalib is biased on downward-looking
imagery. Both predict exactly the same residual pattern.

This settles it with a measurement that contains no gravity estimator at all. Two rigidly
connected sensors see the *same* angular velocity; in their own frames,

    omega_cam(t) = R_cam_imu @ omega_imu(t)

so the relative rotation between two frames can be computed twice — from the images (a
homography between undistorted frames, projected onto SO(3)) and from the gyro (integrated and
rotated by the published extrinsic) — and compared. A mount that is wrong by 14 deg makes every
pair disagree by about 14 deg with a *systematic* axis; a correct mount leaves only the noise of
the flow estimate and the parallax that the homography cannot model.

The test is deliberately not about gravity: it can falsify the published rotation even on
footage where the depicted-gravity estimate is useless.

    python experiments/e_mount_imu.py --seqs indoor_forward_5,indoor_45_2,outdoor_45_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.depicted.mount import kabsch_rotation
from src.depicted.uzh import all_sequences, load_truth, published_mount


def load_imu(members_dir: Path, seq: str):
    """``(t, omega)`` from ``imu.txt``: rows are ``index timestamp wx wy wz ax ay az``."""
    p = Path(members_dir) / seq / "imu.txt"
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    a = np.asarray(rows, dtype=np.float64)
    a = a[np.argsort(a[:, 1])]
    return a[:, 1], a[:, 2:5]


def integrate_gyro(t: np.ndarray, omega: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Rotation vector of the IMU between two times (trapezoid on the gyro)."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return np.zeros(3)
    tt, ww = t[m], omega[m]
    # close the ends so the integral covers exactly [t0, t1]
    if tt[0] > t0:
        tt = np.concatenate([[t0], tt])
        ww = np.vstack([ww[0], ww])
    if tt[-1] < t1:
        tt = np.concatenate([tt, [t1]])
        ww = np.vstack([ww, ww[-1]])
    dt = np.diff(tt)
    return 0.5 * ((ww[:-1] + ww[1:]) * dt[:, None]).sum(axis=0)


def rotation_from_homography(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Closest rotation to ``K^-1 H K`` (SVD projection); assumes the pair is rotation-dominant."""
    A = np.linalg.inv(K) @ H @ K
    A = A / np.cbrt(np.linalg.det(A))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def image_pairs(files, K: np.ndarray, step: int = 1, max_pairs: int = 60):
    """Camera-frame relative rotations from consecutive undistorted frames (KLT + homography)."""
    import cv2

    out = []
    prev = None
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev = None
            continue
        if prev is not None and (i % step == 0):
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01,
                                         minDistance=12, blockSize=7)
            if p0 is not None and len(p0) >= 30:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, p0, None,
                                                     winSize=(21, 21), maxLevel=3)
                ok = (st.ravel() == 1)
                if ok.sum() >= 30:
                    H, mask = cv2.findHomography(p0[ok].reshape(-1, 2),
                                                 p1[ok].reshape(-1, 2), cv2.RANSAC, 3.0)
                    if H is not None and mask is not None and mask.sum() >= 20:
                        R = rotation_from_homography(H, K)
                        if abs(np.degrees(Rotation.from_matrix(R).magnitude())) < 20.0:
                            out.append((i - 1, i, R, float(mask.mean())))
                            if len(out) >= max_pairs:
                                prev = img
                                break
        prev = img
    return out


def error_rotation_stats(meas: list[np.ndarray], pred: list[np.ndarray],
                         mags: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean error rotation between the image and the IMU prediction, with a bootstrap interval.

    For every pair the error is ``E_i = R_img,i @ R_pred,i^T``; a mount error ``D`` shows up as
    ``E_i ~= D`` for *all* pairs, while flow noise shows up as zero-mean scatter around identity.
    So the mean error rotation's magnitude is the mount-error estimate, and its bootstrap
    interval says whether 14 deg is excluded. Averaging in the tangent space is fine here
    because the errors are small.

    Pairs whose own rotation is tiny cannot determine an axis, so the statistic is also reported
    restricted to pairs that rotate by at least 2 deg.
    """
    rv = np.array([Rotation.from_matrix(Rm @ Rp.T).as_rotvec() for Rm, Rp in zip(meas, pred)])
    mags = np.asarray(mags)
    rng = np.random.default_rng(seed)

    def summary(idx):
        if len(idx) < 5:
            return {"n": int(len(idx)), "mean_deg": None, "ci_lo": None, "ci_hi": None}
        m = rv[idx].mean(axis=0)
        vals = []
        for _ in range(n_boot):
            take = rng.integers(0, len(idx), len(idx))
            vals.append(np.linalg.norm(rv[idx][take].mean(axis=0)))
        vals = np.degrees(np.asarray(vals))
        return {"n": int(len(idx)),
                "mean_deg": float(np.degrees(np.linalg.norm(m))),
                "axis": (m / np.linalg.norm(m)).tolist() if np.linalg.norm(m) > 0 else None,
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5))}

    all_idx = np.arange(len(rv))
    big_idx = np.flatnonzero(mags >= 2.0)
    return {"all_pairs": summary(all_idx), "pairs_over_2deg": summary(big_idx),
            "per_pair_error_median_deg": float(np.median(np.degrees(
                np.linalg.norm(rv, axis=1))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uzh")
    ap.add_argument("--members", default="data/uzh_members")
    ap.add_argument("--out", default="results/e_mount")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--max-pairs", type=int, default=60)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    root, members, out = Path(args.root), Path(args.members), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in args.seqs.split(",") if s] or all_sequences()

    recs = []
    for seq in seqs:
        d = root / seq
        if not (d / "meta.json").exists():
            continue
        truth = load_truth(d.parent, d.name)
        R_cam_imu, _ = published_mount(truth.meta)
        R_ci = R_cam_imu.T          # camera <- imu, i.e. the rotation inside T_cam_imu
        K = np.array([[truth.meta["intrinsics_pinhole"]["fx"], 0,
                       truth.meta["intrinsics_pinhole"]["cx"]],
                      [0, truth.meta["intrinsics_pinhole"]["fy"],
                       truth.meta["intrinsics_pinhole"]["cy"]], [0, 0, 1.0]])
        files = sorted((d / "frames_pinhole").glob("*.png"))[: truth.n_frames]
        t_imu, om_imu = load_imu(members, seq)
        pairs = image_pairs(files, K, step=args.step, max_pairs=args.max_pairs)
        pred, meas, angs = [], [], []
        for i, j, R_img, inlier in pairs:
            rv = integrate_gyro(t_imu, om_imu, truth.t_unix[i], truth.t_unix[j])
            R_imu = Rotation.from_rotvec(rv).as_matrix()
            R_pred = R_ci @ R_imu @ R_ci.T
            pred.append(R_pred)
            meas.append(R_img)
            angs.append(float(np.degrees(Rotation.from_matrix(
                R_img @ R_pred.T).magnitude())))
        if len(pred) < 10:
            recs.append({"seq": seq, "skip": f"only {len(pred)} usable pairs"})
            continue
        mags = [float(np.degrees(Rotation.from_matrix(R).magnitude())) for R in meas]
        stats = error_rotation_stats(meas, pred, mags)
        rec = {
            "seq": seq, "rig": truth.rig, "n_pairs": len(pred),
            "pair_rotation_median_deg": float(np.median(mags)),
            "mismatch_median_deg": float(np.median(angs)),
            "mismatch_p90_deg": float(np.percentile(angs, 90)),
            **stats,
            "note": ("mean error rotation between the image-derived and IMU-derived relative "
                     "rotation; a 14 deg mount error would appear here as ~14 deg"),
        }
        recs.append(rec)
        a, b = stats["all_pairs"], stats["pairs_over_2deg"]
        print(f"{seq:18s} {truth.rig:16s} pairs={len(pred):3d} "
              f"pair_rot={np.median(mags):5.2f} deg | mean error "
              f"{a['mean_deg']:5.2f} [{a['ci_lo']:.2f},{a['ci_hi']:.2f}] deg"
              + (f" | over 2 deg: {b['mean_deg']:5.2f} [{b['ci_lo']:.2f},{b['ci_hi']:.2f}]"
                 f" n={b['n']}" if b['mean_deg'] is not None else ""), flush=True)

    (out / "mount_imu.json").write_text(json.dumps(recs, indent=1))
    print(f"\nwrote {out / 'mount_imu.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
