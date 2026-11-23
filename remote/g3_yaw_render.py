"""Track G3: render the 12-clip auxiliary yaw-control slice.

A yaw control literally cannot be composed on the depicted side: the schema lean is the world-up
vector in the body frame, ``u_b = R_bw^T e3``, and a world yaw leaves it unchanged
(``tests/test_g3_yaw_invariance.py``).  The image, however, changes, so the reader's view
dependence can only be tested by rendering.

For each selected clip this app takes Track I's published camera poses
(``/in/i02/clips/<cid>/render.npz``), replaces the world pose with
``Rz(alpha_i) @ R_wc,i`` (translation untouched) with ``alpha_i = min(tilt_req_deg_i, 60)``,
re-renders the same scene with the same settings, encodes H.264 at 30 fps, and writes the
result to this track's volume under ``yaw_slice/<cid>/``.  The clip's truth is unchanged: the
camera is still physical, only looking in another direction.

    pixi run modal run remote/g3_yaw_render.py --clip-ids i02_dev_001,i02_dev_002
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-g3-yaw")
in_vol = modal.Volume.from_name("awmv-i-data", create_if_missing=False)
out_vol = modal.Volume.from_name("awmv-g3-data", create_if_missing=True)

BLENDER_VERSION = "4.5.9"
BLENDER_URL = ("https://download.blender.org/release/Blender4.5/"
               f"blender-{BLENDER_VERSION}-linux-x64.tar.xz")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "xz-utils", "ffmpeg", "libx11-6", "libxi6", "libxxf86vm1",
                 "libxfixes3", "libxrender1", "libgl1", "libsm6", "libgomp1", "libegl1",
                 "libgles2", "libglib2.0-0", "libxkbcommon0", "libxcursor1", "libxrandr2",
                 "libxinerama1", "libdbus-1-3", "libfontconfig1", "libfreetype6")
    .run_commands(
        f"curl -sSL -o /tmp/blender.tar.xz {BLENDER_URL}",
        "mkdir -p /opt && tar -xf /tmp/blender.tar.xz -C /opt && "
        f"ln -s /opt/blender-{BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender",
        "blender --version | head -2",
    )
    .pip_install("numpy<2")
    .env({"BLENDER_EGL": "1"})
    .add_local_file("src/sim/blender_render.py", "/root/blender_render.py", copy=True)
    .add_local_file("configs/g06_yaw_slice.json", "/root/yaw_slice.json", copy=True)
    .add_local_python_source("src")
)


def _rz(alpha: float):
    import numpy as np

    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@app.function(image=image, gpu="L40S", timeout=6 * 3600,
              volumes={"/in": in_vol, "/out": out_vol})
def render(clip_ids: str = "", samples: int = 16, exposure: float = -3.0,
           force: bool = False) -> dict:
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    from src.sim.scenes import layout

    cfg = json.loads(Path("/root/yaw_slice.json").read_text())
    by_id = {c["clip_id"]: c for c in cfg["clips"]}
    ids = [c for c in clip_ids.split(",") if c] or list(by_id)
    rows, failed = [], []
    for cid in ids:
        meta = by_id[cid]
        dst = Path("/out/yaw_slice") / cid
        if (dst / f"{cid}.mp4").is_file() and not force:
            rows.append({"clip_id": cid, "status": "skipped_existing"})
            continue
        try:
            t0 = time.time()
            src = Path("/in/i02/clips") / cid
            npz = dict(np.load(src / "render.npz"))
            poses = np.asarray(npz["poses"], dtype=np.float64).copy()
            tilt = np.asarray(npz["tilt_gt_deg"], dtype=np.float64)
            alpha = np.minimum(tilt, 60.0)
            for i in range(len(poses)):
                poses[i, :3, :3] = _rz(np.radians(alpha[i])) @ poses[i, :3, :3]
            work = Path("/tmp/yaw") / cid
            (work / "frames").mkdir(parents=True, exist_ok=True)
            npz["poses"] = poses
            npz["yaw_control_alpha_deg"] = alpha
            npz["clip_id"] = np.array([cid])
            npz_path = work / "render.npz"
            np.savez_compressed(npz_path, **npz)
            job = {"poses": str(npz_path), "out_dir": str(work / "frames"),
                   "result_json": str(work / "render_timing.json"),
                   "width": 640, "height": 360, "hfov_deg": 90.0,
                   "scene": layout(meta["family"], meta["scene_seed"]),
                   "scene_seed": meta["scene_seed"], "samples": samples, "engine": "EEVEE",
                   "device": "GPU", "exposure": exposure, "denoise": False,
                   "frame_names": [f"{i:04d}.png" for i in range(len(poses))]}
            (work / "job.json").write_text(json.dumps(job))
            p = subprocess.run(["blender", "-b", "-P", "/root/blender_render.py"],
                               capture_output=True, text=True, timeout=5400, cwd="/root",
                               env={**os.environ, "BLENDER_JOB": str(work / "job.json")})
            if p.returncode != 0:
                failed.append({"clip_id": cid, "status": "render_failed",
                               "log": ((p.stdout or "") + (p.stderr or ""))[-1200:]})
                continue
            mp4 = work / f"{cid}.mp4"
            enc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                                  "-i", str(work / "frames" / "%04d.png"),
                                  "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                  "-movflags", "+faststart", str(mp4)],
                                 capture_output=True, text=True, timeout=1200)
            if enc.returncode != 0 or not mp4.is_file():
                failed.append({"clip_id": cid, "status": "encode_failed",
                               "log": (enc.stderr or "")[-1200:]})
                continue
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mp4, dst / f"{cid}.mp4")
            np.savez_compressed(dst / "render.npz", **npz)
            (dst / "yaw_meta.json").write_text(json.dumps({
                "clip_id": cid, "family": meta["family"], "scene_seed": meta["scene_seed"],
                "uptilt_deg": meta["uptilt_deg"], "n_frames": int(len(poses)),
                "alpha_deg_median": float(np.median(alpha)),
                "alpha_deg_max": float(np.max(alpha)),
                "wall_s": round(time.time() - t0, 1),
                "engine": "EEVEE", "samples": samples, "exposure": exposure}, indent=1))
            rows.append({"clip_id": cid, "status": "ok", "frames": int(len(poses)),
                         "wall_s": round(time.time() - t0, 1)})
        except Exception as exc:                      # noqa: BLE001 - reported, never hidden
            failed.append({"clip_id": cid, "status": "exception",
                           "error": f"{type(exc).__name__}: {exc}"[-800:]})
    out_vol.commit()
    return {"rendered": rows, "failed": failed}
