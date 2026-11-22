"""H03: Perspective Fields on the full clips behind the H01/H02/E5 headline numbers.

The pre-registered rule failed, so the affected headline numbers are recomputed with the
independent estimator on every frame, not on the 150-frame sample.  This job reads the clip
frame directories on the existing track volumes and writes one row per frame with the
parameter-head roll/pitch and a reconstructed 3-D world-up direction in camera coordinates.

The reconstruction (validated against the controls where truth is known):

* the field gives, per pixel, the image-plane direction of world-up (x right, and the field's
  y is up-positive, so the image-coordinate vector is (u_x, -u_y));
* the latitude map gives, per pixel, the angle lambda between the ray and the horizon plane;
* with the pixel ray d (camera coordinates) and the in-plane up direction t (unit, orthogonal
  to d), the world-up direction is U = sin(lambda) d + cos(lambda) t, constant over the image.

    set -a; . ./.env; set +a
    pixi run modal run remote/h03_recompute.py::main
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

PF_COMMIT = "d54be737d6eacfb9d39a2b7079a494924b45bb6c"

app = modal.App("awmv-h3-recompute")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml",
                 "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("albumentations==1.3.1", "pyequilib==0.3.0", "timm", "yacs",
                 "scikit-learn", "opencv-contrib-python-headless==4.10.0.84")
    .run_commands(f"pip install --no-cache-dir git+https://github.com/jinlinyi/PerspectiveFields@{PF_COMMIT}")
    .add_local_python_source("src")
)

vol_out = modal.Volume.from_name("awmv-h3-data", create_if_missing=True)
vol_h1 = modal.Volume.from_name("awmv-h-anwm-results", create_if_missing=False)
vol_h2 = modal.Volume.from_name("awmv-h2-results", create_if_missing=False)

# (group, batch, directory that holds one <clip>/frames subdirectory per clip)
ROOTS = [
    ("h03sample", "sample", "/out/h03_frames"),
    ("lingbot_nf4", "probes", "/h1/lingbot"),
    ("solarwm", "probes", "/h2/solarwm/probes"),
    ("solarwm", "grid", "/h2/solarwm/grid"),
    ("solarwm", "seeds", "/h2/solarwm/seeds"),
    ("anwm_seeds", "C", "/h1/clips/C"),
]


def _prep(img):
    import cv2
    h, w = img.shape[:2]
    if h != w:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


def _up_from_field(up, lati, f_px: float = 112.0, size: int = 224, stride: int = 8):
    """Reconstruct one 3-D up vector in camera coordinates from the field + latitude."""
    import numpy as np
    up = np.asarray(up, dtype=np.float64)
    if up.ndim == 3 and up.shape[0] == 2:
        up = up.transpose(1, 2, 0)
    lati = np.asarray(lati, dtype=np.float64)
    if lati.ndim == 3:
        lati = lati[..., 0]
    h, w = up.shape[:2]
    vs, us = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rays = np.stack([(us - cx) / f_px, (vs - cy) / f_px, np.ones_like(us, dtype=float)],
                    axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lam = np.deg2rad(lati[vs, us])
    u_img = np.stack([up[vs, us, 0], -up[vs, us, 1]], axis=-1)   # image coords, y down
    n = np.linalg.norm(u_img, axis=-1, keepdims=True)
    ok = n[..., 0] > 1e-6
    u_img = u_img / np.maximum(n, 1e-9)
    # in-plane (3-D) up direction: the 2-D vector lifted to z=0, orthogonalised to the ray
    t = np.concatenate([u_img, np.zeros_like(u_img[..., :1])], axis=-1)
    t = t - (t * rays).sum(-1, keepdims=True) * rays
    tn = np.linalg.norm(t, axis=-1, keepdims=True)
    keep = ok & (tn[..., 0] > 1e-6)
    if keep.sum() < 10:
        return None
    t = t[keep] / tn[keep]
    r = rays[keep]
    l = lam[keep]
    U = np.sin(l)[:, None] * r + np.cos(l)[:, None] * t
    U /= np.linalg.norm(U, axis=-1, keepdims=True)
    mean = U.mean(axis=0)
    if np.linalg.norm(mean) < 1e-6:
        return None
    return mean / np.linalg.norm(mean)


@app.function(image=image, gpu="A10", timeout=6 * 3600,
              volumes={"/out": vol_out, "/h1": vol_h1, "/h2": vol_h2},
              env={"HF_HOME": "/tmp/hf"})
def read_all(stride: int = 1, limit_clips: int = 0) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from perspective2d import PerspectiveFields

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval().to(device)
    out_dir = Path("/out/h03_recompute")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for group, batch, root in ROOTS:
        rootp = Path(root)
        if not rootp.exists():
            continue
        if group == "h03sample":
            clips = sorted(p for p in rootp.glob("*/*") if any(p.glob("*.png")))
        else:
            clips = sorted(p for p in rootp.glob("*") if (p / "frames").is_dir())
        if limit_clips:
            clips = clips[:limit_clips]
        for clip in clips:
            frames = sorted((clip / "frames").glob("*.png")) if group != "h03sample" \
                else sorted(clip.glob("*.png"))
            for fi, p in enumerate(frames):
                if fi % stride:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                feed = _prep(img)
                with torch.no_grad():
                    pred = pf.inference(img_bgr=feed)
                up3 = _up_from_field(pred["pred_gravity_original"].cpu().numpy(),
                                     pred["pred_latitude_original"].cpu().numpy())
                row = {"group": group, "batch": batch,
                       "clip_id": (clip.parent.name + "/" + clip.name
                                   if group == "h03sample" else clip.name),
                       "frame_index": fi, "frame": p.name,
                       "pf_roll_deg": float(pred["pred_roll"].item()),
                       "pf_pitch_deg": float(pred["pred_pitch"].item()),
                       "pf_vfov_deg": float(pred["pred_general_vfov"].item())}
                if up3 is not None:
                    row["up_cam"] = [float(v) for v in up3]
                    row["up_roll_deg"] = float(np.degrees(np.arctan2(up3[0], -up3[1])))
                rows.append(row)
            print(group, batch, clip.name, len(rows), round(time.time() - t0), flush=True)
    with (out_dir / "pf_rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    vol_out.commit()
    return {"rows": len(rows), "seconds": time.time() - t0}


@app.local_entrypoint()
def main(stride: int = 1, limit_clips: int = 0):
    print(json.dumps(read_all.remote(stride=stride, limit_clips=limit_clips), indent=1))
