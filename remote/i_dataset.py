"""Render the simulated labelled set on Modal (Track I, Step 2).

One Blender pass per clip: the container builds the clip from its frozen row
(``src/sim/dataset.py``), renders the 30 fps frame sequence with EEVEE, encodes H.264, writes
the ground truth at 250 Hz, uploads both to R2 under the track's own prefix and returns the
manifest row with sha256s. Frames stay in the container's /tmp: what ships is the video and
the truth, so the volume does not accumulate 27 GB of PNGs.

Why the container does the upload: the alternative is pulling ~1-2 GB of video through a
workstation with 9.7 GB free, and the sha256 has to be computed on the bytes that actually
reach R2 either way. The R2 keys come from the existing Modal secret ``awmv-r2``.

    pixi run modal run remote/i_dataset.py --clip-ids i02_dev_000,i02_dev_001
    pixi run modal run remote/i_dataset.py --split dev --limit 8
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-i-dataset")
volume = modal.Volume.from_name("awmv-i-data", create_if_missing=True)
r2_secret = modal.Secret.from_name("awmv-r2")

BLENDER_VERSION = "4.5.9"
BLENDER_URL = (f"https://download.blender.org/release/Blender4.5/"
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
    # pandas is not used directly here: `src.verdict` imports its dataset loaders eagerly, and
    # `src.verdict.lean` (the lean definition) lives behind that package import
    .pip_install("numpy<2", "scipy", "pandas", "boto3>=1.34", "python-dotenv>=1.0", "pyyaml")
    .env({"BLENDER_EGL": "1"})
    .add_local_file("src/sim/blender_render.py", "/root/blender_render.py", copy=True)
    .add_local_file("configs/sim_splits_frozen.json", "/root/sim_splits_frozen.json",
                    copy=True)
    .add_local_python_source("src")
)


def _row_ids(table: dict, split: str | None, limit: int, start: int) -> list[str]:
    clips = table["clips"]
    if split:
        clips = [c for c in clips if c["split"] == split]
    ids = [c["clip_id"] for c in clips[start:start + limit]]
    return ids


@app.function(image=image, gpu="L40S", timeout=6 * 3600, volumes={"/vol": volume},
              secrets=[r2_secret])
def render_clips(clip_ids: str = "", split: str = "", limit: int = 0, start: int = 0,
                 samples: int = 16, exposure: float = -3.0, upload: str = "no",
                 keep_frames: bool = False, commit: str = "", force: bool = False) -> dict:
    import hashlib
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    from src.sim.dataset import build_clip, manifest_row
    from src.sim.scenes import layout
    from src.utils import r2 as r2mod

    table = json.loads(Path("/root/sim_splits_frozen.json").read_text())
    ids = [c for c in clip_ids.split(",") if c] or _row_ids(table, split or None, limit or 8, start)
    by_id = {c["clip_id"]: c for c in table["clips"]}
    do_upload = str(upload).lower() in ("1", "true", "yes", "r2")   # off by default: see above
    env = r2mod.load_env() if do_upload else None

    out_rows, timings = [], []
    for cid in ids:
        row = by_id[cid]
        # resume: a clip whose video is already on the volume is never re-rendered
        existing = Path("/vol/i02/clips") / cid / f"{cid}.mp4"
        if existing.is_file() and not force:
            out_rows.append({"clip_id": cid, "status": "skipped_existing"})
            continue
        t0 = time.time()
        clip = build_clip(row)
        work = Path(f"/tmp/render/{cid}")
        (work / "frames").mkdir(parents=True, exist_ok=True)

        # ---- truth and the render bundle
        gt = clip.gt_arrays()
        gt_path = work / "gt.npz"
        np.savez_compressed(gt_path, **gt)
        npz = clip.render_arrays()
        npz["clip_id"] = np.array([cid])
        npz_path = work / "render.npz"
        np.savez_compressed(npz_path, **npz)

        # ---- the job the adapter consumes (same format as the spike's jobs)
        job = {"poses": str(npz_path), "out_dir": str(work / "frames"),
               "result_json": str(work / "render_timing.json"),
               "width": 640, "height": 360, "hfov_deg": 90.0,
               "scene": layout(row["family"], row["scene_seed"]),
               "scene_seed": row["scene_seed"], "samples": samples, "engine": "EEVEE",
               "device": "GPU", "exposure": exposure, "denoise": False,
               "frame_names": [f"{i:04d}.png" for i in range(len(clip.frame_t))]}
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job))

        p = subprocess.run(["blender", "-b", "-P", "/root/blender_render.py"],
                           capture_output=True, text=True, timeout=2400, cwd="/root",
                           env={**os.environ, "BLENDER_JOB": str(job_path)})
        log = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
        if p.returncode != 0:
            out_rows.append({"clip_id": cid, "status": "render_failed",
                             "log_tail": log[-1500:]})
            continue

        # ---- encode H.264 at 30 fps
        mp4 = work / f"{cid}.mp4"
        enc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                              "-i", str(work / "frames" / "%04d.png"),
                              "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(mp4)],
                             capture_output=True, text=True, timeout=1200)
        if enc.returncode != 0 or not mp4.is_file():
            out_rows.append({"clip_id": cid, "status": "encode_failed",
                             "log_tail": (enc.stderr or "")[-1000:]})
            continue

        m = manifest_row(clip)
        m["status"] = "ok"
        m["volume_video"] = f"/vol/i02/clips/{cid}/{cid}.mp4"
        m["volume_gt"] = f"/vol/i02/clips/{cid}/gt.npz"
        m["commit"] = commit
        m["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        split_dir = row["split"]
        key_video = f"track-i/videos/{split_dir}/{row['family']}/{cid}.mp4"
        key_gt = f"track-i/gt/{cid}.npz"
        if do_upload:
            up_v = r2mod.upload(mp4, key_video, env)
            up_g = r2mod.upload(gt_path, key_gt, env)
            if not (up_v["match"] and up_g["match"]):
                out_rows.append({"clip_id": cid, "status": "upload_mismatch",
                                 "up_video": up_v, "up_gt": up_g})
                continue
            m["r2_video"], m["r2_gt"] = key_video, key_gt
            m["sha256_video"], m["sha256_gt"] = up_v["sha256"], up_g["sha256"]
            m["bytes_video"], m["bytes_gt"] = up_v["bytes"], up_g["bytes"]
        else:
            # not published here: r2_* stay empty until experiments/i02_publish.py moves the
            # objects, so the manifest never claims an object that does not exist
            m["sha256_video"] = hashlib.sha256(mp4.read_bytes()).hexdigest()
            m["sha256_gt"] = hashlib.sha256(gt_path.read_bytes()).hexdigest()

        timing = {}
        try:
            timing = json.loads((work / "render_timing.json").read_text())
        except Exception:
            pass
        dt = time.time() - t0
        timings.append({"clip_id": cid, "wall_s": round(dt, 1),
                        "frames": len(clip.frame_t),
                        "fps": round(len(clip.frame_t) / dt, 2),
                        "device_info": timing.get("device_info"),
                        "seconds_per_frame_median": timing.get("seconds_per_frame_median")})
        out_rows.append(m)

        # the volume keeps the small truth and the manifest rows; frames stay in /tmp
        vdir = Path("/vol/i02/clips") / cid
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "manifest_row.json").write_text(json.dumps(m, indent=2))
        (vdir / "timing.json").write_text(json.dumps(timings[-1], indent=2))
        np.savez_compressed(vdir / "gt.npz", **gt)
        np.savez_compressed(vdir / "render.npz", **npz)
        # the video goes to the volume as well: publishing to R2 is done locally, because the
        # Modal secret's key is read-only for this prefix (recorded in docs/decisions.md)
        shutil.copyfile(mp4, vdir / f"{cid}.mp4")
        if keep_frames:
            for f in sorted((work / "frames").glob("*.png")):
                os.replace(f, vdir / f.name)
        volume.commit()

    Path("/vol/i02/last_batch.json").write_text(json.dumps(
        {"rows": out_rows, "timings": timings}, indent=2))
    volume.commit()
    return {"n": len(out_rows), "ok": sum(1 for r in out_rows if r.get("status") == "ok"),
            "timings": timings, "rows": out_rows}
