"""Track C: GeoCalib gravity on prepared UZH clips (real and edited).

Only the depicted-gravity half of the verdict test; the path comes from `remote/c_dpvo.py`
and the fusion happens locally. Output is one JSON per clip with per-frame `gravity_up_cam`.

    pixi run modal run remote/c_verdict.py --seqs indoor_forward_5,indoor_forward_5_tilt_removed
"""
from __future__ import annotations

import modal

app = modal.App("awmv-c-geocalib")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
C_DATA = "/c-data"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)


@app.function(image=image, gpu="A10G", volumes={C_DATA: c_vol}, timeout=3600, cpu=4)
def gravity(seq: str, max_frames: int = 150) -> str:
    """Per-frame up-vector in the camera frame for one prepared UZH variant."""
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    clip = Path(C_DATA) / "uzh" / seq
    meta = json.loads((clip / "meta.json").read_text())
    intr = meta["intrinsics"]
    focal = float(intr["fx"])
    files = sorted((clip / "frames").glob("*.png"))[:max_frames]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(device)
    out = []
    priors = {"focal": torch.tensor(float(focal), device=device)}
    for f in files:
        img = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            res = model.calibrate(t, camera_model="pinhole", priors=priors)
        g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
        out.append({"frame": f.name, "gravity_up_cam": [float(v) for v in g]})
    rec = {"seq": seq, "n": len(out), "focal_used": focal, "rows": out}
    (Path(C_DATA) / "gravity").mkdir(exist_ok=True)
    (Path(C_DATA) / "gravity" / f"{seq}.json").write_text(json.dumps(rec))
    c_vol.commit()
    med = np.median(np.array([r["gravity_up_cam"] for r in out]), axis=0)
    print(json.dumps({"seq": seq, "n": len(out), "median_up_cam": med.tolist()}), flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "", max_frames: int = 150, out_json: str = "") -> None:
    import json
    import os
    recs = []
    for s in [x for x in seqs.split(",") if x]:
        recs.append(json.loads(gravity.remote(s, max_frames)))
        print(f"== gravity {s}: n={recs[-1]['n']}", flush=True)
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        open(out_json, "w").write(json.dumps(recs))
