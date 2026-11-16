"""Track H (H01): depicted-attitude reader for pilot clips (GeoCalib, 224 px, A10G).

Reads frames from the Track H volume (``awmv-h-results``) and writes the same row schema the
ANWM scoring track used (``results/gate_a2/anwm/depicted.jsonl``), so the depicted tilt is
computed by the existing reader and nothing new is invented.

    set -a; . ./.env; set +a
    pixi run modal run remote/h01_gate.py --clip-root lingbot --out-tag lingbot

Frames are resized to 224x224 exactly as ``remote/a_gate.py::anwm_clips`` does.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from src.remote_common import GEOCALIB_COMMIT, VOLUME_MOUNT, ensure_weights, gate_row, make_image

app = modal.App("awmv-h-gate")
image = make_image()
vol_a = modal.Volume.from_name("awmv-a-data", create_if_missing=False)
vol_r = modal.Volume.from_name("awmv-h-results", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: vol_a, "/r": vol_r},
              timeout=3 * 3600)
def depicted(clip_root: str = "clips", out_tag: str = "h01", size: int = 224,
             fov_deg: float = 90.0, limit: int = 0) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    root = Path("/r") / clip_root
    clips = sorted(p for p in root.iterdir() if (p / "frames").is_dir())
    if limit:
        clips = clips[:limit]
    out_dir = Path("/r/h01_depicted") / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    t0 = time.time()
    n_rows = 0
    with (out_dir / "depicted.jsonl").open("w") as fh:
        for clip in clips:
            frames = sorted((clip / "frames").glob("*.png"))
            if not frames:
                frames = sorted((clip / "frames").glob("*.jpg"))
            for p in frames:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                orig = [int(img.shape[1]), int(img.shape[0])]
                if img.shape[0] != size or img.shape[1] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                feed = feed.float().to(device) / 255.0
                with torch.no_grad():
                    res = model.calibrate(feed, camera_model="pinhole",
                                          priors={"focal": torch.tensor(float(f), device=device)})
                fh.write(json.dumps(gate_row(res, {
                    "frame": p.name, "clip_id": clip.name, "frame_std": float(img.std()),
                    "f_prior": float(f), "orig_size": orig})) + "\n")
                n_rows += 1
            print(clip.name, len(frames), flush=True)
    info = {"tag": out_tag, "clip_root": clip_root, "n_clips": len(clips), "n_rows": n_rows,
            "size": size, "f_px": float(f), "seconds": time.time() - t0,
            "geocalib_commit": GEOCALIB_COMMIT, "out": str(out_dir.relative_to("/r"))}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    vol_r.commit()
    return info


@app.local_entrypoint()
def main(clip_root: str = "clips", out_tag: str = "h01", limit: int = 0):
    print(json.dumps(depicted.remote(clip_root, out_tag, limit=limit), indent=2))
