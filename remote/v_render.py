"""Track V, Part 1: re-render the feasible simulated climb clips with vertical-motion defects.

The frozen clip table is read unedited (``configs/sim_splits_frozen.json``, sha256 cdf66e28...);
the defect is applied by ``src/sim/vertical_edits.py`` to the camera's altitude only, and the
camera attitude stays the flatness attitude of the *commanded* path, so the tilt check sees a
feasible clip and only a vertical-motion check can react.

Outputs, one directory per edit on the volume ``awmv-v-data``::

    v_render/<edit_id>/<edit_id>.mp4     H.264 CRF 18, 30 fps, 640x360
    v_render/<edit_id>/gt.npz            t, pos (rendered), pos_commanded, attitudethe and
                                         lean/tilt series, exactly as Track I's gt
    v_render/<edit_id>/meta.json         the edit (kind, factor, z series) + provenance

No R2 upload: the volume is the record, and everything small is pulled from it.

    modal run remote/v_render.py --specs 'i02_dev_005:removed,i02_dev_005:null'
    modal run remote/v_render.py --plan
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-v-render")
volume = modal.Volume.from_name("awmv-v-data", create_if_missing=True)

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
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml")
    .env({"BLENDER_EGL": "1"})
    .add_local_file("src/sim/blender_render.py", "/root/blender_render.py", copy=True)
    .add_local_file("configs/sim_splits_frozen.json", "/root/sim_splits_frozen.json",
                    copy=True)
    .add_local_python_source("src")
)


def plan() -> list[dict]:
    """The render grid: 8 feasible climb clips x 4 vertical defects, 2 nulls, 4 yaw controls."""
    import csv

    rows = list(csv.DictReader(open("results/i02/manifest.csv")))
    climb = sorted(r["clip_id"] for r in rows
                   if r["split"] == "dev" and r["maneuver"] == "climb_turn"
                   and r["label"] == "feasible")
    assert len(climb) == 8, climb
    out = []
    for cid in climb:
        for kind in ("removed", "scaled_0.5", "scaled_1.5", "reversed"):
            out.append({"clip_id": cid, "kind": kind, "edit_id": f"{cid}__{kind}"})
    for cid in climb[:2]:
        out.append({"clip_id": cid, "kind": "null", "edit_id": f"{cid}__null"})
    for cid in climb[:4]:
        out.append({"clip_id": cid, "kind": "yaw", "edit_id": f"{cid}__yaw"})
    return out


@app.function(image=image, gpu="A10G", timeout=6 * 3600, volumes={"/v": volume})
def render(specs: list[dict], force: bool = False, samples: int = 16,
           exposure: float = -3.0) -> str:
    import os
    import shutil
    import subprocess
    import time

    import numpy as np

    from src.sim.dataset import build_clip
    from src.sim.scenes import layout
    from src.sim.vertical_edits import apply_edit, edited_poses

    table = json.loads(Path("/root/sim_splits_frozen.json").read_text())
    by_id = {c["clip_id"]: c for c in table["clips"]}
    rows = []
    for spec in specs:
        cid, kind, eid = spec["clip_id"], spec["kind"], spec["edit_id"]
        vdir = Path("/v/v_render") / eid
        if (vdir / f"{eid}.mp4").is_file() and not force:
            rows.append({"edit_id": eid, "status": "skipped_existing"})
            continue
        t0 = time.time()
        row = by_id[cid]
        clip = build_clip(row)
        pos_commanded = clip.flight.pos.copy()
        meta = apply_edit(clip, kind)
        meta.update({"edit_id": eid, "clip_id": cid, "family": row["family"],
                     "uptilt_deg": float(row["uptilt_deg"]),
                     "scene_seed": int(row["scene_seed"]),
                     "base_label": row["label"], "base_defect": row["defect_family"]})
        work = Path("/tmp/render") / eid
        (work / "frames").mkdir(parents=True, exist_ok=True)
        gt = clip.gt_arrays()
        gt["pos_commanded"] = pos_commanded
        gt["z_commanded_video"] = np.interp(clip.frame_t,
                                            clip.flight.t, pos_commanded[:, 2])
        np.savez_compressed(work / "gt.npz", **gt)
        npz = clip.render_arrays()
        npz["poses"] = edited_poses(clip, kind, meta.get("yaw_deg") or 25.0)
        npz["clip_id"] = np.array([eid])
        np.savez_compressed(work / "render.npz", **npz)
        job = {"poses": str(work / "render.npz"), "out_dir": str(work / "frames"),
               "result_json": str(work / "render_timing.json"),
               "width": 640, "height": 360, "hfov_deg": 90.0,
               "scene": layout(row["family"], row["scene_seed"]),
               "scene_seed": row["scene_seed"], "samples": samples, "engine": "EEVEE",
               "device": "GPU", "exposure": exposure, "denoise": False,
               "frame_names": [f"{i:04d}.png" for i in range(len(clip.frame_t))]}
        (work / "job.json").write_text(json.dumps(job))
        p = subprocess.run(["blender", "-b", "-P", "/root/blender_render.py"],
                           capture_output=True, text=True, timeout=2400, cwd="/root",
                           env={**os.environ, "BLENDER_JOB": str(work / "job.json")})
        if p.returncode != 0:
            rows.append({"edit_id": eid, "status": "render_failed",
                         "log_tail": ((p.stdout or "") + (p.stderr or ""))[-1500:]})
            continue
        mp4 = work / f"{eid}.mp4"
        enc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                              "-i", str(work / "frames" / "%04d.png"),
                              "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(mp4)],
                             capture_output=True, text=True, timeout=1200)
        if enc.returncode != 0 or not mp4.is_file():
            rows.append({"edit_id": eid, "status": "encode_failed",
                         "log_tail": (enc.stderr or "")[-1000:]})
            continue
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mp4, vdir / f"{eid}.mp4")
        shutil.copyfile(work / "gt.npz", vdir / "gt.npz")
        timing = {}
        try:
            timing = json.loads((work / "render_timing.json").read_text())
        except Exception:
            pass
        meta["wall_s"] = round(time.time() - t0, 1)
        meta["n_frames"] = len(clip.frame_t)
        meta["seconds_per_frame_median"] = timing.get("seconds_per_frame_median")
        meta["status"] = "ok"
        (vdir / "meta.json").write_text(json.dumps(meta, indent=1))
        volume.commit()
        rows.append(meta)
    return json.dumps(rows, indent=1)


@app.local_entrypoint()
def main(specs: str = "", all_specs: bool = False, force: bool = False, limit: int = 0):
    if specs:
        todo = []
        for item in specs.split(","):
            cid, kind = item.split(":")
            todo.append({"clip_id": cid, "kind": kind, "edit_id": f"{cid}__{kind}"})
    elif all_specs:
        todo = plan()
    else:
        todo = plan()[:1]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} edits")
    print(render.remote(todo, force))
