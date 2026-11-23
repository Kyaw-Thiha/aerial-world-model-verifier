"""Track N: the frozen GeoCalib-224 pinhole reader on the 200 simulated development clips.

Copied from ``track-l-reader-simreal remote/l_score_sim.py`` (the L01 re-score of Track I's
reader), with only the volume, the input paths and the output paths changed. The reader commit,
weights, camera model, focal prior (90 deg horizontal), stride handling, per-frame estimate and
the ``depicted_lean_from_gravity`` conversion are unchanged; the per-frame record is the same.

    /n-data/sim200/videos/<clip_id>.mp4
    /n-data/sim200/gt/<clip_id>.npz
    -> /n-data/sim200/scores/<clip_id>.npz  and .json

    modal run remote/n_readers.py --clips "..." --stride 2   (see experiments/n05_run_reader.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-n-reader")
volume = modal.Volume.from_name("awmv-n-data", create_if_missing=True)
VOL = "/n-data"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)

WIDTH, HEIGHT = 640, 360


def _summ(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"n": int(x.size), "median": float(np.median(x)),
            "p90": float(np.percentile(np.abs(x), 90)), "min": float(x.min()),
            "max": float(x.max())}


def _conf(res, key):
    import numpy as np

    v = res.get(key) if isinstance(res, dict) else None
    if v is None:
        return float("nan")
    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return float(np.asarray(v, dtype=np.float64).mean())


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=6 * 3600)
def score(clip_ids: str = "", stride: int = 2) -> dict:
    import subprocess
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.verdict.lean import depicted_lean_from_gravity

    ids = [c for c in clip_ids.split(",") if c]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(device)
    e3 = np.array([0.0, 0.0, 1.0])
    f_px = 0.5 * WIDTH / np.tan(np.radians(90.0) / 2.0)

    out_dir = Path(VOL) / "sim200" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for n_done, cid in enumerate(ids, 1):
        done_npz = out_dir / f"{cid}.npz"
        done_json = out_dir / f"{cid}.json"
        if done_npz.exists() and done_json.exists():
            result[cid] = json.loads(done_json.read_text())
            continue
        t0 = time.time()
        mp4 = Path(VOL) / "sim200" / "videos" / f"{cid}.mp4"
        gt_path = Path(VOL) / "sim200" / "gt" / f"{cid}.npz"
        gt = np.load(gt_path)
        gt_hz = float(np.asarray(gt["gt_hz"]).ravel()[0])
        R_bw_cam = np.asarray(gt["R_bw_cam"], dtype=np.float64)
        R_cb = np.asarray(gt["R_cb"], dtype=np.float64)
        lean_dep = np.asarray(gt["lean_dep"], dtype=np.float64)
        lean_req = np.asarray(gt["lean_req"], dtype=np.float64)
        tilt_dep = np.asarray(gt["tilt_dep"], dtype=np.float64)

        frames_dir = Path("/tmp/frames") / cid
        frames_dir.mkdir(parents=True, exist_ok=True)
        for f in frames_dir.glob("*.png"):
            f.unlink()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-vf", f"select=not(mod(n\\,{int(stride)}))", "-vsync", "0",
                        str(frames_dir / "%04d.png")], check=True)
        names = sorted(frames_dir.glob("*.png"))

        rows = []
        for k, p in enumerate(names):
            idx = k * int(stride)
            frame_t = idx / 30.0
            i = int(min(max(round(frame_t * gt_hz), 0), len(tilt_dep) - 1))
            feed = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(feed).permute(2, 0, 1).float().to(device) / 255.0
            priors = {"focal": torch.tensor(float(f_px), device=device)}
            with torch.no_grad():
                res = model.calibrate(t, camera_model="pinhole", priors=priors)
            g = np.asarray(res["gravity"].vec3d.detach().cpu()).ravel()
            g = g / np.linalg.norm(g)
            up_gt = (R_bw_cam[i] @ R_cb).T @ e3
            lean_est = depicted_lean_from_gravity(g[None], R_cb)[0]
            up_body = R_cb @ g
            tilt_est = float(np.degrees(np.arccos(np.clip(up_body @ e3, -1, 1))))
            rows.append({
                "frame": int(idx), "t": float(frame_t), "gt_index": int(i),
                "up_est": [float(v) for v in g], "up_gt": [float(v) for v in up_gt],
                "lean_est_x": float(lean_est[0]), "lean_est_y": float(lean_est[1]),
                "lean_dep_x": float(lean_dep[i, 0]), "lean_dep_y": float(lean_dep[i, 1]),
                "lean_req_x": float(lean_req[i, 0]), "lean_req_y": float(lean_req[i, 1]),
                "tilt_est_deg": tilt_est, "tilt_dep_deg": float(tilt_dep[i]),
                "up_error_deg": float(np.degrees(np.arccos(np.clip(g @ up_gt, -1, 1)))),
                "latitude_confidence_mean": _conf(res, "latitude_confidence"),
                "up_confidence_mean": _conf(res, "up_confidence"),
                "gravity_uncertainty_raw": _conf(res, "gravity_uncertainty"),
            })
        dt = time.time() - t0
        up = np.array([r["up_est"] for r in rows], dtype=np.float64)
        np.savez_compressed(
            out_dir / f"{cid}.npz",
            frame=np.array([r["frame"] for r in rows]),
            t=np.array([r["t"] for r in rows]),
            up_est=up,
            up_gt=np.array([r["up_gt"] for r in rows], dtype=np.float64),
            lean_est=np.array([[r["lean_est_x"], r["lean_est_y"]] for r in rows]),
            lean_dep=np.array([[r["lean_dep_x"], r["lean_dep_y"]] for r in rows]),
            lean_req=np.array([[r["lean_req_x"], r["lean_req_y"]] for r in rows]),
            tilt_est_deg=np.array([r["tilt_est_deg"] for r in rows]),
            tilt_dep_deg=np.array([r["tilt_dep_deg"] for r in rows]),
            up_error_deg=np.array([r["up_error_deg"] for r in rows]),
            latitude_confidence_mean=np.array([r["latitude_confidence_mean"] for r in rows]),
            up_confidence_mean=np.array([r["up_confidence_mean"] for r in rows]),
            gravity_uncertainty_raw=np.array([r["gravity_uncertainty_raw"] for r in rows]),
        )
        err = np.array([r["up_error_deg"] for r in rows])
        summary = {
            "clip_id": cid, "stride": int(stride), "n_frames": len(rows),
            "seconds": round(dt, 1), "fps": round(len(rows) / dt, 2) if dt else None,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "up_error_deg": _summ(err),
            "up_error_signed": None,
            "lean_eq_req_max_abs": float(np.abs(np.array([[r["lean_dep_x"], r["lean_dep_y"]]
                                                          for r in rows])
                                                - np.array([[r["lean_req_x"], r["lean_req_y"]]
                                                            for r in rows])).max()),
        }
        (out_dir / f"{cid}.json").write_text(json.dumps(summary, indent=1))
        if n_done % 10 == 0:
            volume.commit()
        result[cid] = summary
        print(f"[{cid}] {len(rows)} frames {summary['fps']} fps", flush=True)
    volume.commit()
    return result


@app.local_entrypoint()
def main(clips: str = "", shards: int = 4, stride: int = 2):
    ids = [c for c in clips.split(",") if c]
    chunk = max(1, (len(ids) + shards - 1) // shards)
    calls = [score.spawn(",".join(ids[i:i + chunk]), stride) for i in range(0, len(ids), chunk)]
    for c in calls:
        c.get()
