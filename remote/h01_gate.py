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
vol_r = modal.Volume.from_name("awmv-h-gate-results", create_if_missing=True)
vol_anwm = modal.Volume.from_name("awmv-h-anwm-results", create_if_missing=True)
vol_anwm_b = modal.Volume.from_name("awmv-h-anwm-results-b", create_if_missing=True)
vol_anwm_c = modal.Volume.from_name("awmv-h-anwm-results-c", create_if_missing=True)
vol_lingbot = modal.Volume.from_name("awmv-h-lingbot-results", create_if_missing=True)
vol_wan = modal.Volume.from_name("awmv-h-wan-results", create_if_missing=True)
SOURCES = {"anwm": ("/anwm", "clips"), "anwm_b": ("/anwmb", "clips"),
           "anwm_c": ("/anwmc", "clips"), "lingbot": ("/lingbot", "lingbot"),
           "wan": ("/wan", "wan")}


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: vol_a, "/r": vol_r,
                                              "/anwm": vol_anwm, "/anwmb": vol_anwm_b,
                                              "/anwmc": vol_anwm_c,
                                              "/lingbot": vol_lingbot, "/wan": vol_wan}, timeout=3 * 3600)
def depicted(source: str = "anwm", clip_root: str = "", out_tag: str = "h01", size: int = 224,
             fov_deg: float = 90.0, limit: int = 0) -> dict:
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
                # the reader is square: centre-crop a non-square frame first, so the
                # aspect ratio (and therefore every roll angle) is not distorted
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
    info = {"tag": out_tag, "source": source, "clip_root": str(root), "n_clips": len(clips),
            "n_rows": n_rows,
            "size": size, "f_px": float(f), "seconds": time.time() - t0,
            "geocalib_commit": GEOCALIB_COMMIT, "out": str(out_dir.relative_to("/r"))}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    vol_r.commit()
    return info


@app.local_entrypoint()
def main(source: str = "anwm", clip_root: str = "", out_tag: str = "h01", limit: int = 0):
    print(json.dumps(depicted.remote(source, clip_root, out_tag, limit=limit), indent=2))
