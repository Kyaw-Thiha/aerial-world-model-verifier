"""H03: run the learned readers on the pre-registered 150-frame sample.

Two learned readers, both on the *same* feed the depicted-attitude reader uses on generated
frames (square centre crop -> 224x224, pinhole, f = 112 px):

* the reader under test: GeoCalib at commit ``1dedb29`` (pinhole, scalar focal prior);
* the independent estimator: Perspective Fields, Paramnet-360Cities-edina-centered
  (commit ``d54be73``; Adobe Research License, noncommercial research use only, weights are
  downloaded at run time and never committed or re-hosted).

It also runs:
* GeoCalib with the TII/UZH *native* configuration (``simple_divisional`` on the raw
  fisheye) on the 30 real control frames, for context;
* a convention probe: every image rotated by +10 and -10 degrees about its centre, read by
  both estimators, to pin the sign mapping (a pure image rotation is a camera roll).

    set -a; . ./.env; set +a
    pixi run modal run remote/h03_readers.py::main --mode read
    pixi run modal run remote/h03_readers.py::main --mode probe
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from src.remote_common import GEOCALIB_COMMIT

PF_COMMIT = "d54be737d6eacfb9d39a2b7079a494924b45bb6c"
VOL = "awmv-h3-data"

app = modal.App("awmv-h3-readers")
# the Gate 3 perception image, plus the Perspective Fields dependencies, with the local
# ``src`` source added last (Modal refuses build steps after add_local_*)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml",
                 "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .pip_install("albumentations==1.3.1", "pyequilib==0.3.0", "timm", "yacs",
                 "scikit-learn", "opencv-contrib-python-headless==4.10.0.84")
    .run_commands(f"pip install --no-cache-dir git+https://github.com/jinlinyi/PerspectiveFields@{PF_COMMIT}")
    .add_local_python_source("src")
)
vol = modal.Volume.from_name(VOL, create_if_missing=True)


def _prep(img):
    """Square centre crop then 224 px, exactly the reader's generated-frame feed."""
    import cv2
    h, w = img.shape[:2]
    if h != w:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


def _roll(u) -> float:
    import numpy as np
    u = np.asarray(u, dtype=float).ravel()
    return float(np.degrees(np.arctan2(u[0], -u[1])))


@app.function(image=image, gpu="A10", timeout=2 * 3600,
              volumes={"/vol": vol},
              env={"HF_HOME": "/vol/hf"})
def read(tag: str = "main") -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from perspective2d import PerspectiveFields

    from src.remote_common import gate_row

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib(weights="pinhole").to(device)
    pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval().to(device)
    f_px = 112.0
    root = Path("/vol/h03_frames")
    out_dir = Path("/vol/h03_readers")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with (out_dir / f"depicted_{tag}.jsonl").open("w") as fh:
        for p in sorted(root.glob("*/*/*.png")):
            source, clip = p.parts[-3], p.parts[-2]
            raw = cv2.imread(str(p))
            if raw is None:
                continue
            feed = _prep(raw)
            t = torch.from_numpy(cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            t = t.float().to(device) / 255.0
            with torch.no_grad():
                res = gc.calibrate(t, camera_model="pinhole",
                                   priors={"focal": torch.tensor(float(f_px), device=device)})
            row = gate_row(res, {"frame_id": f"{source}/{clip}/{p.stem}", "source": source,
                                 "clip_id": clip, "frame_index": int(p.stem),
                                 "feed": "pinhole224_f112"})
            row["reader_roll_deg"] = _roll(row["gravity_up_cam"])

            # independent estimator on the same 224 feed (its model resizes to its own input)
            with torch.no_grad():
                pred = pf.inference(img_bgr=feed)
            up = pred["pred_gravity_original"].detach().cpu().numpy()
            if up.ndim == 3 and up.shape[0] == 2:
                up = up.transpose(1, 2, 0)
            ux, uy = up[..., 0].ravel(), up[..., 1].ravel()   # field y is up-positive
            ang = np.arctan2(ux, uy)                          # -> image-coord roll
            ang = ang[np.isfinite(ang)]
            row.update({
                "pf_roll_deg": float(pred["pred_roll"].item()),
                "pf_pitch_deg": float(pred["pred_pitch"].item()),
                "pf_vfov_deg": float(pred["pred_general_vfov"].item()),
                "pf_cx": float(pred["pred_rel_cx"].item()),
                "pf_cy": float(pred["pred_rel_cy"].item()),
                "pf_field_roll_deg": float(np.degrees(np.angle(np.mean(np.exp(1j * ang))))),
            })
            # native fisheye reader on the real controls, for context only
            if source == "real_control":
                t2 = torch.from_numpy(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                t2 = t2.float().to(device) / 255.0
                with torch.no_grad():
                    res2 = gc.calibrate(t2, camera_model="simple_divisional")
                g2 = res2["gravity"].vec3d.detach().cpu().numpy().ravel()
                row["reader_fisheye_gravity"] = [float(v) for v in g2]
                row["reader_fisheye_roll_deg"] = _roll(g2)
            fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 25 == 0:
                print(n, time.time() - t0, flush=True)
    info = {"tag": tag, "rows": n, "seconds": time.time() - t0,
            "geocalib_commit": GEOCALIB_COMMIT, "pf_commit": PF_COMMIT}
    (out_dir / f"depicted_{tag}_summary.json").write_text(json.dumps(info, indent=1))
    vol.commit()
    return info


@app.function(image=image, gpu="A10", timeout=1800, volumes={"/vol": vol},
              env={"HF_HOME": "/vol/hf"})
def probe(seed_frames: int = 9) -> dict:
    """Sign convention: rotate the image by +10 and -10 deg, read both estimators."""
    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from perspective2d import PerspectiveFields

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib(weights="pinhole").to(device)
    pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval().to(device)
    root = Path("/vol/h03_frames")
    picks = []
    for source in ("real_control", "sim_control", "anwm", "solarwm"):
        picks += sorted(root.glob(f"{source}/*/*.png"))[::37][:seed_frames // 4 + 1]
    rows = []
    for p in picks:
        source, clip = p.parts[-3], p.parts[-2]
        raw = cv2.imread(str(p))
        feed = _prep(raw)
        for deg in (-10.0, 0.0, 10.0):
            M = cv2.getRotationMatrix2D((feed.shape[1] / 2, feed.shape[0] / 2), deg, 1.0)
            rot = cv2.warpAffine(feed, M, (feed.shape[1], feed.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            t = torch.from_numpy(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            t = t.float().to(device) / 255.0
            with torch.no_grad():
                res = gc.calibrate(t, camera_model="pinhole",
                                   priors={"focal": torch.tensor(112.0, device=device)})
                pred = pf.inference(img_bgr=rot)
            g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
            rows.append({"frame_id": f"{source}/{clip}/{p.stem}", "rotation_deg": deg,
                         "reader_roll_deg": _roll(g),
                         "pf_roll_deg": float(pred["pred_roll"].item())})
    out = Path("/vol/h03_readers")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation_probe.json").write_text(json.dumps(rows, indent=1))
    vol.commit()
    return {"n": len(rows), "frames": len(picks)}


@app.local_entrypoint()
def main(mode: str = "read"):
    if mode == "read":
        print(json.dumps(read.remote(), indent=1))
    else:
        print(json.dumps(probe.remote(), indent=1))
