"""H03: assemble the pre-registered 150-frame sample.

Reads only local/decoded frames and ground truth; runs no estimator.  The sample and the
frame-index rule are fixed in ``configs/h03_reader_validity_prereg.yaml``.

    pixi run -e gpu python experiments/h03_select_frames.py

Writes ``data/h03_frames/<source>/<clip>/<idx>.png`` (gitignored) and
``results/h03/frames_manifest.csv`` (committed), one row per frame, with the ground-truth
world-up direction in camera coordinates where it is known.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("/home/kevin/Documents/Projects/aerial-world-model-verifier")
OUT = ROOT / "data/h03_frames"
RESULTS = ROOT / "results/h03"
TMP = Path("/tmp/h3data")
N_PER_CLIP = 5

ANWM = ["C_cruise_ctl_course_0_s1", "C_slalom_5_fixed_0_s1", "C_slalom_10_fixed_0_s1",
        "C_climb_turn_10_fixed_0_s1", "C_climb_turn_20_fixed_0_s1",
        "C_accel_brake_20_course_0_s1"]
LINGBOT_NF4 = ["probe_level", "probe_rollramp_p20", "probe_rollramp_m20"]
LINGBOT_BF16 = ["probe_level", "probe_rollramp_fast20", "probe_rollramp_m20"]
SOLARWM_PROBES = ["probe_level", "probe_p20", "probe_m20"]
SOLARWM_GRID = ["grid_302OLP89E75X7MFPRR58QG7CJPZCA7_processed_015_turn10_flatness",
                "grid_302OLP89E75X7MFPRR58QG7CJPZCA7_processed_015_turn20_mirrored_roll",
                "grid_302OLP89E75X7MFPRR58QG7CJPZCA7_processed_015_slalom10_flatness"]
TRACK_I = ["i02_dev_000", "i02_dev_001", "i02_dev_002", "i02_dev_004", "i02_dev_005",
           "i02_dev_006"]
TII = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-13a-trackRATM",
       "flight-01p-ellipse", "flight-07p-lemniscate", "flight-13p-trackRATM"]


def idx_rule(n: int) -> list[int]:
    return [int(round(v)) for v in np.linspace(0, n - 1, N_PER_CLIP)]


def _write_png(img, dst: Path) -> None:
    import cv2
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


def _read_frames_mp4(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def _row(source, clip, idx, n, path, scene, up_cam=None, note="") -> dict:
    up = ["" if up_cam is None else f"{float(v):.9f}" for v in (up_cam if up_cam is not None
                                                               else [np.nan] * 3)]
    return {"frame_id": f"{source}/{clip}/{idx:03d}", "source": source, "clip_id": clip,
            "frame_index": idx, "n_frames": n, "path": path, "scene": scene,
            "up_cam_x": up[0], "up_cam_y": up[1], "up_cam_z": up[2], "note": note}


def from_png_dir(rows, source, clip, src_dir: Path, path="", scene="", note=""):
    files = sorted(Path(src_dir).glob("*.png"))
    n = len(files)
    for i in idx_rule(n):
        dst = OUT / source / clip / f"{i:03d}.png"
        _write_png(__import__("cv2").imread(str(files[i])), dst)
        rows.append(_row(source, clip, i, n, path, scene, note=note))


def from_mp4(rows, source, clip, mp4: Path, path="", scene="", note="", ups=None):
    frames = _read_frames_mp4(mp4)
    n = len(frames)
    for i in idx_rule(n):
        dst = OUT / source / clip / f"{i:03d}.png"
        _write_png(frames[i], dst)
        rows.append(_row(source, clip, i, n, path, scene,
                         None if ups is None else ups[i], note=note))


def track_i_truth(clip: str) -> np.ndarray:
    d = np.load(TMP / "i" / clip / "render.npz", allow_pickle=True)
    Rwc = np.asarray(d["poses"], dtype=np.float64)[:, :3, :3]
    return np.einsum("nji,j->ni", Rwc, np.array([0.0, 0.0, 1.0]))


def tii_truth(flight: str, n_frames: int) -> tuple[list[int], np.ndarray]:
    """Ground-truth up in camera at the 5 sampled frames of one flight."""
    import sys
    sys.path.insert(0, str(ROOT))
    from scipy.spatial.transform import Rotation

    from src.datasets.tii import load_tii
    from src.datasets.tii_video import load_tii_video
    from src.perception.frames import camera_to_body_pitched, up_in_camera

    data = SHARED / "data/raw"
    clip = load_tii_video(data / "tii_video", flight, mocap_dir=data / "tii",
                          calib_dir=data / "tii_video/calibration")
    control = "autonomous" if flight.split("-")[1].endswith("a") else "piloted"
    epoch_us = int(load_tii(data / "tii" / control / f"mocap_{flight}.csv").meta["epoch_start_us"])
    files = sorted((data / "tii_frames" / flight).glob("*.jpg"))
    ts = np.array([int(f.stem.split("_")[1]) for f in files], dtype=np.int64)
    t = (ts - epoch_us).astype(np.float64) * 1e-6
    rv = Rotation.from_quat(clip.quat).as_rotvec()
    tt = np.clip(t, clip.t[0], clip.t[-1])
    rv_t = np.column_stack([np.interp(tt, clip.t, rv[:, k]) for k in range(3)])
    R_bw = Rotation.from_rotvec(rv_t).as_matrix()
    R_cb = camera_to_body_pitched(np.deg2rad(clip.cam_uptilt_deg))
    ups = up_in_camera(R_bw, R_cb)
    idx = idx_rule(len(files))
    return idx, ups[idx]


def main() -> int:
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # ANWM: PNG frames on the h-anwm volume, pulled to /tmp/h3data/a
    for c in ANWM:
        from_png_dir(rows, "anwm", c, TMP / "a" / c / "frames",
                     path=c.replace("C_", "").rsplit("_s", 1)[0], scene="anwm_city")

    # LingBot NF4 / bf16: PNG frame dirs in the shared checkout
    for c in LINGBOT_NF4:
        from_png_dir(rows, "lingbot_nf4", c, SHARED / "data/h01_lingbot_out" / c / "frames",
                     path=c, scene="lingbot")
    for c in LINGBOT_BF16:
        from_png_dir(rows, "lingbot_bf16", c, SHARED / "data/h02_lingbot_full_out" / c / "frames",
                     path=c, scene="lingbot")

    # SolarWM: probes have PNG frames, grid clips are decoded from mp4
    probe_meta = {}
    for c in SOLARWM_PROBES:
        from_png_dir(rows, "solarwm", c, SHARED / "data/h02_solarwm_probes_out" / c / "frames",
                     path=c, scene="anwm_city")
        meta = json.loads((SHARED / "data/h02_solarwm_probes_out" / c / "meta.json").read_text())
        probe_meta[c] = meta
    for c in SOLARWM_GRID:
        from_mp4(rows, "solarwm", c, SHARED / "data/h02_solarwm_grid_out" / c / "clip.mp4",
                 path=c.split("_processed_015_")[-1], scene="anwm_city")

    # Track I simulated controls, truth from render.npz
    for c in TRACK_I:
        ups = track_i_truth(c)
        row = json.loads((TMP / "i" / c / "manifest_row.json").read_text())
        from_mp4(rows, "sim_control", c, TMP / "i" / c / f"{c}.mp4",
                 path=row["maneuver"], scene=row["family"], ups=ups,
                 note=f"mount={row['uptilt_deg']}deg peak_tilt_req={row['peak_tilt_req_deg']:.2f}")

    # TII real controls, truth from mocap + analytic mount
    for flight in TII:
        files = sorted((SHARED / "data/raw/tii_frames" / flight).glob("*.jpg"))
        n = len(files)
        idx, ups = tii_truth(flight, n)
        for k, i in enumerate(idx):
            dst = OUT / "real_control" / flight / f"{i:03d}.png"
            _write_png(cv2.imread(str(files[i])), dst)
            rows.append(_row("real_control", flight, i, n, flight.split("-", 2)[2],
                             "tii", ups[k]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "frames_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    (RESULTS / "frames_manifest_summary.json").write_text(json.dumps(
        {"total": len(rows), "by_source": by_src, "per_clip": N_PER_CLIP,
         "frame_index_rule": "round(linspace(0, n-1, 5))"}, indent=1))
    print(json.dumps({"total": len(rows), "by_source": by_src}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
