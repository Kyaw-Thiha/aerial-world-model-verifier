"""Phase 1 camera solve: validate intrinsics and camera-to-body rotation on pixels.

The mocap gives the camera pose from the *outside*, so a static arena becomes a
"structure from known motion" problem: track points across a window, triangulate
with the mocap poses, and measure reprojection error. The reprojection error is
flat in the true camera-to-body rotation and grows steeply away from it, so

* the residual at the released ``drone_to_camera.json`` rotation says whether the
  mount/extrinsic is right;
* a local 3-parameter solve around it gives the mount rotation the pixels
  actually support, and the difference is **reported**, never applied silently;
* the same residual validates the fisheye model and the virtual pinhole view —
  a warped undistortion shows up as a large floor even at the best rotation.

    pixi run -e gpu python experiments/phase1_camera_solve.py --fov 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.datasets.tii_video import camera_to_body, flight_parts, load_calibration
from src.perception import undistort as U

ALL_FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def frame_ts(name: str) -> int:
    return int(Path(name).stem.split("_")[1])


def load_window(frames_dir: Path, t_lo: float, t_hi: float, epoch_us: int,
                vc: U.VirtualCamera, max_frames: int = 12) -> tuple[list, np.ndarray]:
    import cv2

    names = sorted(frames_dir.glob("*.jpg"))
    ts = np.array([(frame_ts(n.name) - epoch_us) * 1e-6 for n in names])
    sel = np.flatnonzero((ts >= t_lo) & (ts <= t_hi))
    if len(sel) > max_frames:
        sel = sel[np.linspace(0, len(sel) - 1, max_frames).astype(int)]
    imgs, times = [], []
    for i in sel:
        raw = cv2.imread(str(names[i]))
        imgs.append(U.undistort(raw, vc))
        times.append(ts[i])
    return imgs, np.array(times)


def track_points(imgs: list, max_pts: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track corners from the first frame across the window.

    Returns ``pts (T, N, 2)`` with NaN where a point was lost, plus the mask of
    points observed in at least 3 frames.
    """
    import cv2

    g0 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    mask = (imgs[0].sum(axis=2) > 30).astype(np.uint8) * 255  # ignore black borders
    pts0 = cv2.goodFeaturesToTrack(g0, max_pts, 0.01, 8, mask=mask)
    if pts0 is None:
        raise RuntimeError("no corners")
    pts0 = np.asarray(pts0, dtype=np.float64).reshape(-1, 1, 2)
    T = len(imgs)
    pts = np.full((T, len(pts0), 2), np.nan)
    pts[0] = pts0[:, 0, :]
    alive = np.ones(len(pts0), dtype=bool)
    for t in range(1, T):
        p = pts[t - 1].copy()
        idx = np.flatnonzero(alive & ~np.isnan(p[:, 0]))
        if len(idx) == 0:
            break
        prev = cv2.cvtColor(imgs[t - 1], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(imgs[t], cv2.COLOR_BGR2GRAY)
        new, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p[idx].astype(np.float32).reshape(-1, 1, 2), None,
                                              winSize=(21, 21), maxLevel=3,
                                              criteria=(cv2.TERM_CRITERIA_EPS
                                                        | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        back, st2, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, np.asarray(new, dtype=np.float32).reshape(-1, 1, 2), None, winSize=(21, 21),
                                                maxLevel=3,
                                                criteria=(cv2.TERM_CRITERIA_EPS
                                                          | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        new = np.asarray(new, dtype=np.float64).reshape(-1, 2)
        back = np.asarray(back, dtype=np.float64).reshape(-1, 2)
        fb = np.linalg.norm(back - p[idx], axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        pts[t, idx[good]] = new[good]
        alive[idx[~good]] = False
    obs = (~np.isnan(pts[:, :, 0])).sum(axis=0)
    return pts, obs >= 3, obs


def project(pts_w: np.ndarray, R_wc: np.ndarray, C_w: np.ndarray, K: np.ndarray) -> np.ndarray:
    X = pts_w - C_w
    xc = X @ R_wc            # R_wc^T @ X per row
    z = xc[:, 2]
    return np.column_stack([K[0, 0] * xc[:, 0] / z + K[0, 2],
                            K[1, 1] * xc[:, 1] / z + K[1, 2]]), z


def triangulate(pts: np.ndarray, R_wc: np.ndarray, C_w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation, vectorised over points.

    For each view the ray direction d_t and centre C_t give rows ``[M_t | -M_t C_t]``
    with ``M_t = I - d_t d_t^T``; the point is the null vector of the stacked
    system, i.e. the smallest eigenvector of the accumulated 4x4 ``A^T A``.
    """
    T, N, _ = pts.shape
    ok = ~np.isnan(pts[:, :, 0])                                     # (T, N)
    u = np.where(ok, pts[:, :, 0], K[0, 2])
    v = np.where(ok, pts[:, :, 1], K[1, 2])
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                     np.ones_like(u)], axis=-1)          # (T, N, 3)
    d = np.einsum("tij,tnj->tni", R_wc, rays)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    M = np.eye(3)[None, None] - d[..., :, None] * d[..., None, :]   # (T, N, 3, 3)
    MC = np.einsum("tnij,tj->tni", M, C_w)                          # (T, N, 3)
    okf = ok.astype(np.float64)
    M = M * okf[..., None, None]
    MC = MC * okf[..., None]
    A = np.zeros((N, 4, 4))
    A[:, :3, :3] = np.einsum("tnij->nij", M)
    A[:, :3, 3] = -np.einsum("tni->ni", MC)
    A[:, 3, :3] = A[:, :3, 3]
    A[:, 3, 3] = np.einsum("tni,tni->n", MC, MC)
    _, vecs = np.linalg.eigh(A)
    h = vecs[:, :, 0]                                                # smallest eigenvalue
    with np.errstate(invalid="ignore", divide="ignore"):
        X = h[:, :3] / h[:, 3:4]
    X[~np.isfinite(X).all(axis=1)] = np.nan
    return X


def reprojection_rms(pts: np.ndarray, X: np.ndarray, R_wc: np.ndarray, C_w: np.ndarray,
                     K: np.ndarray) -> tuple[float, int]:
    errs = []
    for t in range(len(R_wc)):
        ok = ~np.isnan(pts[t, :, 0]) & np.isfinite(X).all(axis=1)
        if not ok.any():
            continue
        px, z = project(X[ok], R_wc[t], C_w[t], K)
        good = z > 0.1
        if not good.any():
            continue
        errs.append(np.linalg.norm(px[good] - pts[t, ok][good], axis=1))
    if not errs:
        return float("nan"), 0
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2))), int(len(e))


def evaluate(R_cb, R_bw, C_w, pts, K):
    R_wc = np.einsum("nij,jk->nik", R_bw, R_cb)
    X = triangulate(pts, R_wc, C_w, K)
    rms, n = reprojection_rms(pts, X, R_wc, C_w, K)
    return float(rms), X


def solve_rotation(R_cb0: np.ndarray, R_bw: np.ndarray, C_w: np.ndarray, pts: np.ndarray,
                   K: np.ndarray, k0: np.ndarray) -> dict:
    def cost(w):
        R = Rotation.from_rotvec(w).as_matrix() @ R_cb0
        rms, _ = evaluate(R, R_bw, C_w, pts, K)
        return rms if np.isfinite(rms) else 1e6

    res = minimize(cost, k0, method="Powell",
                   options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 300})
    R_opt = Rotation.from_rotvec(res.x).as_matrix() @ R_cb0
    return {"rotvec_delta": res.x.tolist(), "rms": float(res.fun),
            "delta_deg": float(np.degrees(np.linalg.norm(res.x))),
            "R_opt": R_opt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", default="data/raw/tii_video")
    ap.add_argument("--frames-root", default="data/raw/tii_frames")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--window", type=float, nargs=2, default=[0.3, 1.6],
                    help="seconds into the flight used for the solve")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = Path("results/phase1_camera_solve") / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"fov_deg": args.fov, "window_s": args.window, "max_frames": args.max_frames,
           "frames_root": args.frames_root, "flights": args.flights or ALL_FLIGHTS}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    res_all: dict = {"run_id": run_id, "git": git_hash(), "config": cfg, "flights": {}}

    for flight in (args.flights or [f for v in ALL_FLIGHTS.values() for f in v]):
        control, track = flight_parts(flight)
        cal = load_calibration(Path(args.video_root) / "calibration", control, track)
        ext = camera_to_body(Path(args.video_root) / "calibration", control, track)
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        vc = U.virtual_camera(cal["K"], cal["dist"], (640, 480), fov_deg=args.fov)
        imgs, times = load_window(Path(args.frames_root) / flight, args.window[0],
                                 args.window[1], clip.meta["epoch_start_us"], vc,
                                 args.max_frames)
        if len(imgs) < 3:
            print(f"{flight}: not enough frames in window")
            continue
        R_bw_all = Rotation.from_quat(clip.quat).as_matrix()
        idx = [int(np.argmin(np.abs(clip.t - t))) for t in times]
        R_bw = R_bw_all[idx]
        C_w = clip.pos[idx]
        baseline = float(np.max(np.linalg.norm(C_w - C_w[0], axis=1)))
        pts, keep, obs = track_points(imgs)
        pts = pts[:, keep]
        rec = {"n_frames": len(imgs), "n_tracked": int(keep.sum()),
               "baseline_m": baseline, "window_s": [float(times[0]), float(times[-1])],
               "source_speed_mps": float(np.linalg.norm(
                   np.gradient(clip.pos, clip.t, axis=0)[idx].mean(axis=0)))}
        for name, key in (("json", "R_json"), ("jsonT", "R_json_T")):
            rms, _ = evaluate(ext[key], R_bw, C_w, pts, vc.K)
            rec[f"rms_{name}"] = rms
            sol = solve_rotation(ext[key], R_bw, C_w, pts, vc.K, np.zeros(3))
            rec[f"solved_{name}"] = {"rms": sol["rms"], "delta_deg": sol["delta_deg"],
                                     "rotvec_delta": sol["rotvec_delta"]}
            if name == "json":
                best = sol
        # mount offset supported by the pixels, relative to the json rotation
        d = best["R_opt"] @ ext["R_json"].T
        rv = Rotation.from_matrix(d).as_rotvec()
        rec["mount_offset_vs_json"] = {"rotvec_deg": np.degrees(rv).tolist(),
                                      "angle_deg": float(np.degrees(np.linalg.norm(rv)))}
        res_all["flights"][flight] = rec
        print(f"{flight}: tracked={keep.sum()} base={baseline:.2f}m "
              f"rms_json={rec['rms_json']:.2f} rms_jsonT={rec['rms_jsonT']:.2f} "
              f"solved={best['rms']:.2f} offset={rec['mount_offset_vs_json']['angle_deg']:.2f}deg "
              f"(metadata uptilt {clip.cam_uptilt_deg})", flush=True)

    (out / "metrics.json").write_text(json.dumps(res_all, indent=2, default=float))
    (out / "git.txt").write_text(git_hash() + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
