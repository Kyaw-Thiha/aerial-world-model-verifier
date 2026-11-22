"""Track H2: depicted-attitude reader for SolarWM and full-precision LingBot outputs.

Same reader and row schema as the ANWM/H01 tracks (GeoCalib 224 px pinhole, gravity ->
body attitude, square centre crop). Reads frames from the H2 results volume.

    set -a; . ./.env; set +a
    pixi run modal run remote/h02_gate.py::main --source lingbot_full --out-tag lingbot-full
    pixi run modal run remote/h02_gate.py::main --source solarwm --clip-root test --out-tag solarwm-test
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from src.remote_common import GEOCALIB_COMMIT, VOLUME_MOUNT, ensure_weights, gate_row, make_image

app = modal.App("awmv-h2-gate")
image = make_image()
vol_a = modal.Volume.from_name("awmv-a-data", create_if_missing=False)
vol_r = modal.Volume.from_name("awmv-h2-results", create_if_missing=True)
SOURCES = {"lingbot_full": ("/r", "lingbot_full"), "solarwm": ("/r", "solarwm")}


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: vol_a, "/r": vol_r},
              timeout=3 * 3600)
def depicted(source: str = "lingbot_full", clip_root: str = "", out_tag: str = "h02",
             size: int = 224, fov_deg: float = 90.0, limit: int = 0,
             stride: int = 1) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    mount, default_sub = SOURCES[source]
    root = Path(mount) / (clip_root or default_sub)
    clips = sorted(p for p in root.iterdir() if (p / "frames").is_dir())
    if limit:
        clips = clips[:limit]
    out_dir = Path("/r/depicted") / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    t0 = time.time()
    n_rows = 0
    with (out_dir / "depicted.jsonl").open("w") as fh:
        for clip in clips:
            frames = sorted((clip / "frames").glob("*.png"))[::max(1, int(stride))]
            for p in frames:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                orig = [int(img.shape[1]), int(img.shape[0])]
                h, w = img.shape[:2]
                if h != w:
                    side = min(h, w)
                    y0, x0 = (h - side) // 2, (w - side) // 2
                    img = img[y0:y0 + side, x0:x0 + side]
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
    info = {"tag": out_tag, "source": source, "clip_root": str(root), "stride": stride,
            "n_clips": len(clips),
            "n_rows": n_rows, "size": size, "f_px": float(f),
            "seconds": time.time() - t0, "geocalib_commit": GEOCALIB_COMMIT,
            "out": str(out_dir.relative_to("/r"))}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    vol_r.commit()
    return info


@app.local_entrypoint()
def main(source: str = "lingbot_full", clip_root: str = "", out_tag: str = "h02",
         limit: int = 0, stride: int = 1):
    print(json.dumps(depicted.remote(source, clip_root, out_tag, limit=limit, stride=stride),
                     indent=2))
