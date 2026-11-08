"""Phase 1 camera check: calibration diagnostics, extrinisics, overlays.

Verifies the two things that every later number depends on, before any estimator
is blamed:

1. **Calibration** — what the released model can represent (invertible radius,
   corner coverage, fx/fy anisotropy) and how badly it warps straight structure.
2. **Camera-to-body rotation** — the mocap attitude plus the mount rotation
   predicts where world-up and the horizon project. That prediction is drawn on
   the undistorted frame and checked against the scene, and the constant offset
   between the predicted up point and the dominant vertical vanishing point is
   reported per flight. Nothing is fitted away.

    pixi run -e gpu python experiments/phase1_camera_check.py
    pixi run -e gpu python experiments/phase1_camera_check.py --fov 90 --frames 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.datasets.tii import load_tii
from src.datasets.tii_video import camera_to_body, flight_parts, load_calibration
from src.perception import frames as F
from src.perception import undistort as U

FLIGHTS = ["flight-01a-ellipse", "flight-01a-ellipse"]
ALL_FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}
CANDIDATES = {"json": "R_json", "jsonT": "R_json_T"}


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def frame_ts(name: str) -> int:
    """``00012_1691756159816137.jpg`` -> 1691756159816137 (Unix microseconds)."""
    return int(Path(name).stem.split("_")[1])


def vvp_ransac(p0: np.ndarray, p1: np.ndarray, n_iter: int = 400, thresh_deg: float = 1.0,
               seed: int = 0) -> dict:
    """Dominant vanishing point of 2D segments, with angular inlier count.

    Angles are measured in a frame where the principal point is the origin scaled
    by the focal length, so a one-pixel error is one unit; the "angle" between a
    segment and the vanishing point is the distance from the segment's supporting
    line to the point, in pixels scaled by the focal length.
    """
    rng = np.random.default_rng(seed)
    d = p1 - p0
    norm = np.linalg.norm(d, axis=1)
    keep = norm > 1e-6
    p0, d, norm = p0[keep], d / norm[:, None], norm[keep]
    if len(p0) < 4:
        return {"ok": False, "n_segments": int(len(p0))}
    m = 0.5 * (p0 + p1[keep])
    best = None
    for _ in range(n_iter):
        i, j = rng.choice(len(p0), 2, replace=False)
        A = np.array([p0[i], p0[j]])
        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            continue
        vp = np.linalg.solve(A, np.ones(2))
        lines = np.stack([p0, p1[keep]], axis=1)  # not used; keep memory small
        num = np.abs(d[:, 0] * (vp[1] - p0[:, 1]) - d[:, 1] * (vp[0] - p0[:, 0]))
        res = num  # normalised direction, so this is point-line distance in px
        inl = res < thresh_deg
        score = int(inl.sum())
        if best is None or score > best["score"]:
            best = {"score": score, "vp": vp, "inlier_mask": inl}
    return {"ok": True, "n_segments": int(len(p0)), **best}


def overlay_frame(img: np.ndarray, vc: U.VirtualCamera, up_cam: np.ndarray,
                  label: str) -> tuple[np.ndarray, dict]:
    import cv2

    px, ok = F.project_direction(up_cam[None], vc.K)
    out = img.copy()
    hz = F.horizon_points(up_cam, vc.K)
    if len(hz) > 1:
        cv2.polylines(out, [hz.astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1,
                      cv2.LINE_AA)
    info = {"up_visible": bool(ok[0]) and 0 <= px[0, 0] < vc.out_size[0]
            and 0 <= px[0, 1] < vc.out_size[1],
            "up_px": [float(px[0, 0]), float(px[0, 1])], "up_in_front": bool(ok[0]),
            "horizon_points_in_image": int(np.sum((hz[:, 0] >= 0) & (hz[:, 0] < vc.out_size[0])
                                                  & (hz[:, 1] >= 0) & (hz[:, 1] < vc.out_size[1])))
            if len(hz) else 0}
    if ok[0]:
        x, y = int(round(px[0, 0])), int(round(px[0, 1]))
        cv2.circle(out, (np.clip(x, -200, 800), np.clip(y, -200, 700)), 6, (0, 0, 255), 2)
        cv2.putText(out, "up", (np.clip(x + 8, 0, 600), np.clip(y, 18, 470)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out, info


def straightness(img: np.ndarray) -> dict:
    """Long-structure survival metric: number and length of LSD segments."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(gray)[0]
    if segs is None or len(segs) == 0:
        return {"n": 0, "n_long": 0, "max_len": 0.0, "p95_len": 0.0}
    segs = np.asarray(segs, dtype=np.float64)
    if segs.ndim == 2:                     # OpenCV 5 returns (N, 4)
        segs = segs.reshape(-1, 2, 2)
    lens = np.linalg.norm(segs[:, 0, :] - segs[:, 1, :], axis=1)
    return {"n": int(len(lens)), "n_long": int((lens > 40).sum()),
            "max_len": float(lens.max()), "p95_len": float(np.percentile(lens, 95)),
            "p0": segs[:, 0, :], "p1": segs[:, 1, :], "len": lens}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", default="data/raw/tii_video")
    ap.add_argument("--frames-root", default="data/raw/tii_frames")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--out-size", type=int, nargs=2, default=[640, 480])
    ap.add_argument("--frames", type=int, default=6, help="frames per flight for the overlays")
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = Path("results/phase1_camera") / run_id
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    flights = args.flights or [f for v in ALL_FLIGHTS.values() for f in v]

    cfg = {"fov_deg": args.fov, "out_size": args.out_size, "frames_per_flight": args.frames,
           "video_root": args.video_root, "frames_root": args.frames_root,
           "mocap_root": args.mocap_root, "flights": flights,
           "virtual_camera": "square pixels, same optical axis as source"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    metrics: dict = {"run_id": run_id, "git": git_hash(), "config": cfg, "flights": {}}
    for flight in flights:
        control, track = flight_parts(flight)
        cal = load_calibration(Path(args.video_root) / "calibration", control, track)
        ext = camera_to_body(Path(args.video_root) / "calibration", control, track)
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        R_bw = F.body_to_world(clip.quat)
        diag = U.model_diagnostics(cal["K"], cal["dist"], (640, 480))
        vc = U.virtual_camera(cal["K"], cal["dist"], (640, 480), fov_deg=args.fov,
                              out_size=tuple(args.out_size))

        fdir = Path(args.frames_root) / flight
        names = sorted(fdir.glob("*.jpg"))
        step = max(1, len(names) // args.frames)
        picks = names[::step][: args.frames]

        rec = {"diagnostics": diag,
               "virtual_camera": {"valid_fraction": vc.valid_fraction,
                                  "max_source_theta_deg": vc.max_source_theta_deg,
                                  "focal_px": float(vc.K[0, 0])},
               "uptilt_metadata_deg": clip.cam_uptilt_deg,
               "extrinsic_key": ext["key"], "frames": []}
        for cand_name, cand_key in CANDIDATES.items():
            R_cb = ext[cand_key]
            up = F.up_in_camera(R_bw, R_cb)
            el = np.degrees(np.arcsin(np.clip(
                np.einsum("nij,j->ni", R_bw, R_cb[:, 2])[:, 2], -1, 1)))
            rec[f"elevation_{cand_name}"] = {"median_deg": float(np.median(el)),
                                             "p10_deg": float(np.percentile(el, 10)),
                                             "p90_deg": float(np.percentile(el, 90))}

        import cv2

        for name in picks:
            ts = frame_ts(name.name)
            idx = int(np.argmin(np.abs(clip.t - (ts - clip.meta["epoch_start_us"]) * 1e-6)))
            t_rel = (ts - clip.meta["epoch_start_us"]) * 1e-6
            speed = float(np.linalg.norm(np.gradient(clip.pos, clip.t, axis=0)[idx]))
            raw = cv2.imread(str(name))
            und = U.undistort(raw, vc)
            entry = {"frame": name.name, "t_rel_s": t_rel, "speed_mps": speed,
                     "straightness_raw": {k: v for k, v in straightness(raw).items()
                                          if not isinstance(v, np.ndarray)},
                     "straightness_undistorted": {k: v for k, v in straightness(und).items()
                                                  if not isinstance(v, np.ndarray)}}
            for cand_name, cand_key in CANDIDATES.items():
                R_cb = ext[cand_key]
                up = F.up_in_camera(R_bw[idx][None], R_cb)[0]
                img, info = overlay_frame(und, vc, up, f"{flight} {t_rel:.2f}s "
                                                         f"v={speed:.1f} {cand_name}")
                cv2.imwrite(str(out / "overlays" / f"{flight}_{cand_name}_{name.stem}.png"), img)
                entry[f"overlay_{cand_name}"] = info
            # vanishing point of long segments, for later comparison with `up`
            st = straightness(und)
            if st["n"]:
                longm = st["len"] > 40
                vp = vvp_ransac(st["p0"][longm], st["p1"][longm], seed=0, thresh_deg=1.5)
                if vp.get("ok"):
                    entry["vp_ransac"] = {"vp_px": vp["vp"].tolist(),
                                          "inliers": vp["score"],
                                          "n_long": int(longm.sum())}
                    for cand_name, cand_key in CANDIDATES.items():
                        u = F.up_in_camera(R_bw[idx][None], ext[cand_key])[0]
                        px, _ = F.project_direction(u[None], vc.K)
                        entry[f"up_{cand_name}_px"] = px[0].tolist()
                        entry[f"vp_minus_up_{cand_name}_px"] = (
                            vp["vp"] - px[0]).tolist()
            rec["frames"].append(entry)
        metrics["flights"][flight] = rec
        print(f"{flight}: fov_valid={vc.valid_fraction:.2f} "
              f"max_theta={vc.max_source_theta_deg:.1f} "
              f"elev_json={rec['elevation_json']['median_deg']:.1f} "
              f"elev_jsonT={rec['elevation_jsonT']['median_deg']:.1f} "
              f"(metadata {clip.cam_uptilt_deg})", flush=True)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "git.txt").write_text(git_hash() + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
