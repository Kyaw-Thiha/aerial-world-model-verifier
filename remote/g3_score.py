"""Track G3: GeoCalib reader at stride 1 on the simulated clips, plus the yaw slice.

Reads Track I's published clip bundles (``awmv-i-data``: ``i02/clips/<cid>/`` with
``render.npz`` and the H.264 video) **read-only**, and writes only to this track's volume
(``awmv-g3-data``).  Nothing is written to the Track I volume or prefix.

The reader is the frozen instrument: GeoCalib, pinhole weights, known focal prior
(f = 0.5 * 640 / tan(45 deg)), the same settings as ``remote/i_score.py``.  The per-frame
rows use the same field names as Track I's score files so the stride-1 versus stride-2
consistency check is a direct comparison.

    pixi run modal run remote/g3_score.py --clip-ids i02_dev_000,i02_dev_001 --out scores/stride1
    pixi run modal run remote/g3_score.py --clip-ids-from /tmp/ids.txt --out scores/yaw_slice \
        --root yaw_slice
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-g3-score")
in_vol = modal.Volume.from_name("awmv-i-data", create_if_missing=False)
out_vol = modal.Volume.from_name("awmv-g3-data", create_if_missing=True)

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


def _run(clip_ids: str = "", out: str = "scores/stride1",
         base: str = "/in/i02/clips", batch: int = 1,
         stride: int = 1, force: bool = False) -> dict:
    import subprocess
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.verdict.lean import depicted_lean_from_gravity

    import os

    # make our own runs reproducible: GeoCalib on an A10 is not bit-deterministic run to run
    # (measured: median |d lean| 2.1e-3, max 5.5e-2 between two identical runs)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    ids = [c for c in clip_ids.split(",") if c]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(device)
    e3 = np.array([0.0, 0.0, 1.0])
    outdir = Path("/out") / out
    outdir.mkdir(parents=True, exist_ok=True)

    done, failed = [], []
    for cid in ids:
        target = outdir / f"{cid}_score.json"
        if target.is_file() and not force:
            done.append(cid)
            continue
        cdir = Path(base) / cid
        try:
            npz = np.load(cdir / "render.npz")
            poses = npz["poses"]
            R_cb = npz["R_cb"]
            lean_req = npz["lean_req"]
            tilt_gt = npz["tilt_gt_deg"]
            mp4 = cdir / f"{cid}.mp4"
            frames_dir = Path("/tmp/frames") / cid
            frames_dir.mkdir(parents=True, exist_ok=True)
            for f in frames_dir.glob("*.png"):
                f.unlink()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                 "-vf", f"select=not(mod(n\\,{int(stride)}))", "-vsync", "0",
                 str(frames_dir / "%04d.png")], check=True)
            names = sorted(frames_dir.glob("*.png"))
            rows = []
            t0 = time.time()
            t_dec = 0.0
            t_inf = 0.0
            f_px = 0.5 * 640 / np.tan(np.radians(90.0) / 2.0)
            b = max(1, int(batch))
            for k0 in range(0, len(names), b):
                if k0 * int(stride) >= len(poses):
                    break
                ta = time.time()
                feeds, idxs = [], []
                for k in range(k0, min(k0 + b, len(names))):
                    idx = k * int(stride)
                    if idx >= len(poses):
                        break
                    feed = cv2.cvtColor(cv2.imread(str(names[k])), cv2.COLOR_BGR2RGB)
                    feeds.append(torch.from_numpy(feed).permute(2, 0, 1).float())
                    idxs.append(idx)
                t_dec += time.time() - ta
                tb = time.time()
                tt = torch.stack(feeds).to(device) / 255.0
                with torch.no_grad():
                    res = model.calibrate(tt, camera_model="pinhole",
                                          priors={"focal": torch.full((len(feeds),), float(f_px),
                                                                      device=device)})
                gs = np.asarray(res["gravity"].vec3d.detach().cpu())
                t_inf += time.time() - tb
                for j, idx in enumerate(idxs):
                    g = gs[j] / np.linalg.norm(gs[j])
                    up_gt = poses[idx, :3, :3].T @ e3
                    lean_est = depicted_lean_from_gravity(g[None], R_cb)[0]
                    up_body = R_cb @ g
                    rows.append({
                        "frame": int(idx),
                        "up_error_deg": float(np.degrees(np.arccos(np.clip(g @ up_gt, -1, 1)))),
                        "tilt_est_deg": float(np.degrees(np.arccos(np.clip(up_body @ e3, -1, 1)))),
                        "tilt_gt_deg": float(tilt_gt[idx]),
                        "lean_est_x": float(lean_est[0]), "lean_est_y": float(lean_est[1]),
                        "lean_req_x": float(lean_req[idx, 0]),
                        "lean_req_y": float(lean_req[idx, 1])})
            dt = time.time() - t0
            le = np.array([[r["lean_est_x"], r["lean_est_y"]] for r in rows])
            lr = np.array([[r["lean_req_x"], r["lean_req_y"]] for r in rows])
            payload = {
                "clip_id": cid, "stride": int(stride),
                "n_frames": len(rows), "seconds": round(dt, 1), "batch": int(batch),
                "decode_s": round(t_dec, 1), "infer_s": round(t_inf, 1),
                "fps": round(len(rows) / dt, 2) if dt else None,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "up_error_deg": _summ([r["up_error_deg"] for r in rows]),
                "tilt_error_deg": _summ([r["tilt_est_deg"] - r["tilt_gt_deg"] for r in rows]),
                "lean_slope_on_req": _slope(le, lr),
                "lean_corr_on_req": _corr(le, lr),
                "rows": rows}
            target.write_text(json.dumps(payload))
            done.append(cid)
        except Exception as exc:                      # noqa: BLE001 - reported, never hidden
            failed.append({"clip_id": cid, "error": f"{type(exc).__name__}: {exc}"[-500:]})
    out_vol.commit()
    return {"out": out, "scored": len(done), "failed": failed,
            "ids": done if len(done) <= 5 else None}


@app.function(image=image, gpu="A10G", timeout=4 * 3600,
              volumes={"/in": in_vol, "/out": out_vol})
def score(clip_ids: str = "", out: str = "scores/stride1", base: str = "/in/i02/clips",
          batch: int = 1, force: bool = False) -> dict:
    return _run(clip_ids=clip_ids, out=out, base=base, batch=batch, force=force)


@app.function(image=image, gpu="A100-80GB", timeout=4 * 3600,
              volumes={"/in": in_vol, "/out": out_vol})
def score_a100(clip_ids: str = "", out: str = "scores/stride1", base: str = "/in/i02/clips",
               batch: int = 1, force: bool = False) -> dict:
    return _run(clip_ids=clip_ids, out=out, base=base, batch=batch, force=force)


@app.function(image=image, gpu="L40S", timeout=4 * 3600,
              volumes={"/in": in_vol, "/out": out_vol})
def score_l40s(clip_ids: str = "", out: str = "scores/stride1", base: str = "/in/i02/clips",
               batch: int = 1, force: bool = False) -> dict:
    return _run(clip_ids=clip_ids, out=out, base=base, batch=batch, force=force)


@app.function(image=image, gpu="H100", timeout=4 * 3600,
              volumes={"/in": in_vol, "/out": out_vol})
def score_h100(clip_ids: str = "", out: str = "scores/stride1", base: str = "/in/i02/clips",
               batch: int = 8, force: bool = False) -> dict:
    return _run(clip_ids=clip_ids, out=out, base=base, batch=batch, force=force)
