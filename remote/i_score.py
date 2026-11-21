"""Score rendered clips with the project's tilt reader (Track I, Step 2/3).

This is the first number Step 3 asks for: the reader's error on simulated frames against
exact truth, before any detection rate. It reads a clip from the volume -- the H.264 video,
the 250 Hz truth and the render-side lean -- decodes the frames, runs GeoCalib exactly as the
project runs it on real footage (pinhole model, known focal prior), and reports:

* up-vector error (the reader's own error, in degrees);
* depicted tilt against the rendered truth and against the path's requirement;
* the depicted-on-required lean slope per axis (the gain statistic read early).

The reader is the frozen instrument; nothing here tunes it. Scoring is restricted to
development scenes until the development pass is closed and committed (pre-registration,
`evaluation.heldout_pass`); the function refuses a held-out clip unless `allow_heldout` is
passed, and that flag is for the single, final held-out pass.

    pixi run modal run remote/i_score.py --clip-ids i02_dev_000,i02_dev_001
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-i-score")
volume = modal.Volume.from_name("awmv-i-data", create_if_missing=True)
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)


def _summ(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"n": int(x.size), "median": float(np.median(x)),
            "p90": float(np.percentile(np.abs(x), 90)), "min": float(x.min()),
            "max": float(x.max())}


@app.function(image=image, gpu="A10G", volumes={"/vol": volume}, timeout=3600)
def score(clip_ids: str = "", camera_model: str = "pinhole", focal_prior: bool = True,
          stride: int = 1, allow_heldout: bool = False) -> dict:
    import subprocess
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.verdict.lean import depicted_lean_from_gravity

    ids = [c for c in clip_ids.split(",") if c]
    if not ids:
        return {"error": "no clip ids"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole" if camera_model == "pinhole" else "distorted").to(device)
    e3 = np.array([0.0, 0.0, 1.0])

    out = {}
    for cid in ids:
        cdir = Path("/vol/i02/clips") / cid
        man = cdir / "manifest_row.json"
        if man.is_file():
            split = json.loads(man.read_text())["split"]
            if split == "heldout" and not allow_heldout:
                out[cid] = {"error": "held-out clip: the development pass is not closed"}
                continue
        npz = np.load(cdir / "render.npz")
        poses = npz["poses"]
        R_cb = npz["R_cb"]
        lean_gt = npz["lean_gt"]
        lean_req = npz["lean_req"]
        tilt_gt = npz["tilt_gt_deg"]
        width, height = 640, 360
        # the video is the artifact that ships; the frames are decoded from it, not from PNGs
        mp4 = cdir / f"{cid}.mp4"
        frames_dir = Path("/tmp/frames") / cid
        frames_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in frames_dir.glob("*.png"):
            f.unlink()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-vf", f"select=not(mod(n\\,{int(stride)}))", "-vsync", "0",
                        str(frames_dir / "%04d.png")], check=True)
        names = sorted(frames_dir.glob("*.png"))
        rows = []
        t0 = time.time()
        for k, p in enumerate(names):
            idx = k * int(stride)
            if idx >= len(poses):
                break
            feed = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(feed).permute(2, 0, 1).float().to(device) / 255.0
            f_px = 0.5 * width / np.tan(np.radians(90.0) / 2.0)
            priors = {"focal": torch.tensor(float(f_px), device=device)} if focal_prior else None
            with torch.no_grad():
                res = model.calibrate(t, camera_model=camera_model, priors=priors)
            g = np.asarray(res["gravity"].vec3d.detach().cpu()).ravel()
            g = g / np.linalg.norm(g)
            up_gt = poses[idx, :3, :3].T @ e3
            lean_est = depicted_lean_from_gravity(g[None], R_cb)[0]
            up_body = R_cb @ g
            tilt_est = float(np.degrees(np.arccos(np.clip(up_body @ e3, -1, 1))))
            rows.append({"frame": int(idx),
                         "up_error_deg": float(np.degrees(np.arccos(np.clip(g @ up_gt, -1, 1)))),
                         "tilt_est_deg": tilt_est,
                         "tilt_gt_deg": float(tilt_gt[idx]),
                         "lean_est_x": float(lean_est[0]), "lean_est_y": float(lean_est[1]),
                         "lean_req_x": float(lean_req[idx, 0]),
                         "lean_req_y": float(lean_req[idx, 1])})
        dt = time.time() - t0
        le = np.array([[r["lean_est_x"], r["lean_est_y"]] for r in rows])
        lr = np.array([[r["lean_req_x"], r["lean_req_y"]] for r in rows])
        terr = np.array([r["tilt_est_deg"] - r["tilt_gt_deg"] for r in rows])
        out[cid] = {
            "n_frames": len(rows), "seconds": round(dt, 1),
            "fps": round(len(rows) / dt, 2) if dt else None,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "up_error_deg": _summ([r["up_error_deg"] for r in rows]),
            "tilt_error_deg": _summ(terr),
            "lean_slope_on_req": _slope(le, lr),
            "lean_corr_on_req": _corr(le, lr),
            "tilt_est_median_deg": float(np.median([r["tilt_est_deg"] for r in rows])),
            "tilt_gt_median_deg": float(np.median([r["tilt_gt_deg"] for r in rows])),
            "rows": rows,
        }
        (Path("/vol/i02/scores")).mkdir(parents=True, exist_ok=True)
        (Path("/vol/i02/scores") / f"{cid}_score.json").write_text(json.dumps(out[cid], indent=2))
        volume.commit()
    return {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}


def _slope(est, req) -> dict:
    import numpy as np

    out = {}
    for k, name in enumerate(("forward", "left")):
        x = np.asarray(req, dtype=float)[:, k]
        y = np.asarray(est, dtype=float)[:, k]
        ok = np.isfinite(x) & np.isfinite(y)
        denom = float((x[ok] ** 2).sum())
        out[name] = float((x[ok] * y[ok]).sum() / denom) if denom > 1e-12 else None
    return out


def _corr(a, b) -> dict:
    import numpy as np

    out = {}
    for k, name in enumerate(("forward", "left")):
        x = np.asarray(b, dtype=float)[:, k]
        y = np.asarray(a, dtype=float)[:, k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
            out[name] = None
        else:
            out[name] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out
