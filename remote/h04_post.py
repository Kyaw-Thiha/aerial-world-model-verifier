"""H04: reader and yaw-control passes for the continuous-attitude campaign.

``depicted``: the frozen reader (GeoCalib 1dedb29, 224 px square pinhole, f = 112 px) on every
frame of every generated clip on the results volume, same feed as H01/H02.

``warp_yaw``: the yaw control.  Arm A, draw 0: each frame warped by the camera-frame rotation
about world up of +20 deg, built from the commanded poses with the project's own edit code
(``src.negatives.edits.condition_maps`` + ``src.depicted.edits.warp``, pinhole f = 240 px).
A world-yaw rotation leaves the tilt unchanged, so the statistic must not move.

    set -a; . ./.env; set +a
    pixi run modal run remote/h04_post.py::main --mode yaw
    pixi run modal run remote/h04_post.py::main --mode read
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from src.remote_common import GEOCALIB_COMMIT, make_image

app = modal.App("awmv-h3-post")
vol_r = modal.Volume.from_name("awmv-h3-results", create_if_missing=True)
image = make_image()


@app.function(image=image, cpu=8, timeout=3600, volumes={"/r": vol_r})
def warp_yaw(tag: str = "main", yaw_deg: float = 20.0, f_px: float = 240.0,
             width: int = 864, height: int = 480) -> dict:
    import cv2
    import numpy as np

    from src.depicted.edits import warp
    from src.generation.attitude import body_rotation
    from src.negatives.edits import condition_maps
    from src.perception.frames import camera_to_body_pitched

    root = Path("/r/solarwm") / tag
    poses = json.loads((Path("/r/h04") / "yaw_poses.json").read_text())
    K = np.array([[f_px, 0.0, (width - 1) / 2.0], [0.0, f_px, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]])
    R_cb = camera_to_body_pitched(0.0)
    out_root = Path("/r/solarwm") / f"{tag}_yaw{yaw_deg:+.0f}"
    done = 0
    for clip in sorted(root.iterdir()):
        if not clip.name.endswith("_A_flatness_g0"):
            continue
        name = clip.name[len("main_"):-len("_A_flatness_g0")]
        p = poses[name]
        yaw = np.deg2rad(np.asarray(p["yaw"]))
        roll = np.asarray(p["roll"])
        pitch = np.asarray(p["pitch"])
        R_bw = body_rotation(yaw, pitch, roll)
        R_cw = np.einsum("nij,jk->nik", R_bw, R_cb)
        M = condition_maps("yaw_control", R_bw, R_cw,
                           yaw_rot_rad=np.deg2rad(yaw_deg))
        d = out_root / clip.name / "frames"
        d.mkdir(parents=True, exist_ok=True)
        frames = sorted((clip / "frames").glob("*.png"))
        for k, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None or k >= len(M):
                continue
            out, valid = warp(img, M[k], "pinhole", K)
            cv2.imwrite(str(d / fp.name), out)
        done += 1
        print(clip.name, flush=True)
    (out_root / "WARP.json").write_text(json.dumps(
        {"tag": tag, "yaw_deg": yaw_deg, "f_px": f_px, "size": [width, height],
         "clips": done}))
    vol_r.commit()
    return {"clips": done, "out": str(out_root)}


@app.function(image=image, gpu="A10", timeout=4 * 3600, volumes={"/r": vol_r},
              env={"HF_HOME": "/tmp/hf"})
def depicted(tag: str = "main", size: int = 224, fov_deg: float = 90.0,
             stride: int = 1, shard: int = 0, shards: int = 1) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.remote_common import gate_row

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(device)
    root = Path("/r/solarwm") / tag
    clips = sorted(p for p in root.iterdir() if (p / "frames").is_dir())[shard::shards]
    out_dir = Path("/r/h04") / f"depicted_{tag}_s{shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    t0 = time.time()
    n = 0
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
                if img.shape[0] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                feed = feed.float().to(device) / 255.0
                with torch.no_grad():
                    res = model.calibrate(feed, camera_model="pinhole",
                                          priors={"focal": torch.tensor(float(f), device=device)})
                row = gate_row(res, {"frame": p.name, "clip_id": clip.name,
                                     "frame_std": float(img.std()),
                                     "orig_size": orig, "tag": tag})
                fh.write(json.dumps(row) + "\n")
                n += 1
            print(clip.name, len(frames), flush=True)
    info = {"tag": tag, "shard": shard, "shards": shards, "root": str(root),
            "n_clips": len(clips), "n_rows": n,
            "size": size, "f_px": float(f), "stride": stride,
            "seconds": time.time() - t0, "geocalib_commit": GEOCALIB_COMMIT}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
    vol_r.commit()
    return info


@app.local_entrypoint()
def main(mode: str = "read", tag: str = "main", shard: int = 0, shards: int = 1):
    if mode == "yaw":
        print(json.dumps(warp_yaw.remote(tag=tag), indent=1))
    else:
        print(json.dumps(depicted.remote(tag=tag, shard=shard, shards=shards), indent=1))
