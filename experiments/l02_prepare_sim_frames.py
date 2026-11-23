"""Track L, L02: extract the simulated training frames and their exact labels.

The training label is the world-up direction in the camera frame, computed from the render pose
that produced the pixels -- never from an estimate:

    up_cam[k] = R_cb^T R_bw_cam[i_k]^T e3,  i_k = round((k / 30) * gt_hz)

with ``k`` the video frame index and ``R_bw_cam``, ``R_cb`` from the published truth npz. The
construction is checked against the published ``lean_dep`` and ``tilt_dep`` arrays of the same
frames; a clip whose check fails is not written and the run exits non-zero.

Frames are decoded from the published video (the artifact that ships), spread evenly over the
flight, and stored as JPEG at quality 92 under ``data/l02/frames/<clip_id>/<frame>.jpg`` with a
per-clip ``labels.json``. Only feasible development clips are read; nothing held out is touched.

    PYTHONPATH=. python experiments/l02_prepare_sim_frames.py --per-clip 40
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.verdict.lean import depicted_lean_from_gravity

MANIFEST = Path("results/i02/manifest.csv")
VIDEOS = Path("data/l01/sim_dev/videos")
GT = Path("data/l01/sim_dev/gt")
OUT = Path("data/l02/frames")
E3 = np.array([0.0, 0.0, 1.0])


def up_cam_for(gt_path: Path, frame_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    z = np.load(gt_path)
    R = np.asarray(z["R_bw_cam"], dtype=np.float64)
    R_cb = np.asarray(z["R_cb"], dtype=np.float64)
    hz = float(np.asarray(z["gt_hz"]).ravel()[0])
    t = np.asarray(z["t"], dtype=np.float64)
    lean = np.asarray(z["lean_dep"], dtype=np.float64)
    tilt = np.asarray(z["tilt_dep"], dtype=np.float64)
    # the render schedule of src/sim/dataset.py: frame_t = k / 30, index = round(frame_t * gt_hz)
    scale = float(np.asarray(z["gt_hz"]).ravel()[0])

    def idx_for(k):
        return int(min(max(round((k / 30.0) * scale), 0), len(t) - 1))

    idx = np.array([idx_for(int(k)) for k in frame_idx], dtype=int)
    up = np.einsum("ji,nj->ni", R_cb, np.einsum("nji,j->ni", R[idx], E3))
    lean_chk = depicted_lean_from_gravity(up, R_cb)
    tilt_chk = np.degrees(np.arccos(np.clip(np.einsum("ij,nj->ni", R_cb, up) @ E3, -1, 1)))
    check = {"lean_max_abs": float(np.abs(lean_chk - lean[idx]).max()),
             "tilt_max_abs": float(np.abs(tilt_chk - tilt[idx]).max()),
             "gt_hz": hz}
    return up, check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    out = Path(args.out)
    rows = [r for r in csv.DictReader(MANIFEST.open())
            if r["split"] == "dev" and r["label"] == "feasible"]
    manifest = {"clips": 0, "frames": 0, "per_clip_frames": args.per_clip,
                "label": "up_cam = R_cb^T R_bw_cam^T e3 at the video frame", "clip_list": []}
    worst = 0.0
    for r in rows:
        cid = r["clip_id"]
        m = int(r["n_frames"])
        k = args.per_clip
        picks = np.unique(np.round(np.linspace(0, m - 1, min(k, m))).astype(int))
        up, check = up_cam_for(GT / f"{cid}.npz", picks)
        worst = max(worst, check["lean_max_abs"], check["tilt_max_abs"])
        if check["lean_max_abs"] > 1e-9 or check["tilt_max_abs"] > 1e-9:
            raise SystemExit(f"label check failed for {cid}: {check}")
        src = VIDEOS / f"{cid}.mp4"
        tmp = Path("/tmp/l02_frames") / cid
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-fps_mode", "passthrough", str(tmp / "%05d.jpg")], check=True)
        decoded = sorted(tmp.glob("*.jpg"))
        if len(decoded) < m:
            raise SystemExit(f"{cid}: decoded {len(decoded)} frames, expected {m}")
        dst = out / cid
        dst.mkdir(parents=True, exist_ok=True)
        frames = []
        for j, fi in enumerate(picks):
            target = dst / f"{int(fi):05d}.jpg"
            shutil.copyfile(decoded[int(fi)], target)
            frames.append({"frame": int(fi), "t": float(int(fi) / 30.0),
                           "up_cam": [float(v) for v in up[j]]})
        (dst / "labels.json").write_text(json.dumps({
            "clip_id": cid, "family": r["family"], "scene_seed": int(r["scene_seed"]),
            "uptilt_deg": float(r["uptilt_deg"]), "maneuver": r["maneuver"],
            "video": str(src.relative_to(Path("."))),
            "label_check": check, "frames": frames}, indent=0))
        manifest["clips"] += 1
        manifest["frames"] += len(frames)
        manifest["clip_list"].append({"clip_id": cid, "family": r["family"],
                                      "scene_seed": int(r["scene_seed"]),
                                      "frames": len(frames)})
        shutil.rmtree(tmp)
        if manifest["clips"] % 10 == 0:
            print(f"{manifest['clips']} clips, {manifest['frames']} frames", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {manifest['clips']} clips, {manifest['frames']} frames, "
          f"worst label error {worst:.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
