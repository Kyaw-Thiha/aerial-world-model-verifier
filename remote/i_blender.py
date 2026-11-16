"""Blender render on Modal (Track I spike, candidate D): headless cloud GPU, measured.

The local spike already renders on an RTX 3050 laptop; this app answers the criteria that a
laptop cannot: does the same adapter run **headless on a rented GPU**, and what is the
throughput per GPU-hour (which sets the dataset cost).

Blender is downloaded in the image (its own CUDA/OptiX kernels are bundled), so no display,
no driver install beyond the container's, and no GPU-specific build.

    pixi run modal run remote/i_blender.py::render --tag i01_feasible
"""
from __future__ import annotations

import modal

app = modal.App("awmv-i-blender")
volume = modal.Volume.from_name("awmv-i-data", create_if_missing=True)

BLENDER_VERSION = "4.5.9"
BLENDER_URL = (f"https://download.blender.org/release/Blender4.5/"
               f"blender-{BLENDER_VERSION}-linux-x64.tar.xz")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "xz-utils", "libx11-6", "libxi6", "libxxf86vm1", "libxfixes3",
                 "libxrender1", "libgl1", "libsm6", "libgomp1", "libegl1", "libgles2",
                 "libglib2.0-0", "libxkbcommon0", "libxcursor1", "libxrandr2",
                 "libxinerama1", "libdbus-1-3", "libfontconfig1", "libfreetype6")
    .run_commands(
        f"curl -sSL -o /tmp/blender.tar.xz {BLENDER_URL}",
        "mkdir -p /opt && tar -xf /tmp/blender.tar.xz -C /opt && "
        f"ln -s /opt/blender-{BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender",
        "blender --version | head -3",
    )
    .pip_install("numpy", "pillow")
    .env({"BLENDER_EGL": "1"})
    .add_local_python_source("src")
)


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=3600)
def benchmark(tag: str = "i01_feasible", n_frames: int = 8) -> dict:
    """Throughput per renderer setting on this GPU, on the first ``n_frames`` frames.

    This is the number that sets the dataset cost: frames per GPU-hour times the frame
    count of the planned set. Every configuration renders the *same* first frames so the
    comparison is not confounded by scene content.
    """
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np

    base = json.loads(Path(f"/vol/spike/{tag}_job.json").read_text())
    npz = np.load(f"/vol/spike/{tag}_poses.npz")
    sub = {k: (npz[k][:n_frames] if k in ("poses", "yaw", "tilt_gt_deg", "lean_gt",
                                          "lean_req", "frame_t") else npz[k]) for k in npz.files}
    np.savez_compressed(f"/vol/spike/{tag}_poses_bench.npz", **sub)
    results = {}
    configs = [
        {"engine": "CYCLES", "samples": 8, "denoise": True},
        {"engine": "CYCLES", "samples": 16, "denoise": True},
        {"engine": "CYCLES", "samples": 48, "denoise": True},
        {"engine": "EEVEE", "samples": 16, "denoise": False},
    ]
    for cfg in configs:
        job = dict(base)
        job.update({"poses": f"/vol/spike/{tag}_poses_bench.npz",
                    "out_dir": f"/vol/frames_bench/{tag}_{cfg['engine']}_{cfg['samples']}",
                    "result_json": None, "exposure": -3.0, "device": "GPU",
                    "frame_names": [f"{i:04d}.png" for i in range(n_frames)], **cfg})
        jp = Path(f"/tmp/bench_{cfg['engine']}_{cfg['samples']}.json")
        jp.write_text(json.dumps(job))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "src.sim.blender_render", str(jp)],
                           capture_output=True, text=True, timeout=1500, cwd="/root")
        log = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
        key = f"{cfg['engine']}_{cfg['samples']}"
        entry = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1),
                 "log_tail": log[-1200:]}
        for line in log.splitlines():
            if line.startswith("RENDER_RESULT "):
                entry["render"] = json.loads(line[len("RENDER_RESULT "):])
        results[key] = entry
        volume.commit()
    return results


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=3600)
def render(tag: str = "i01_feasible", samples: int = 48, engine: str = "CYCLES",
           denoise: bool = True, limit: int = 0) -> dict:
    import json
    import subprocess
    import sys
    from pathlib import Path

    job = json.loads(Path(f"/vol/spike/{tag}_job.json").read_text())
    job["poses"] = f"/vol/spike/{tag}_poses.npz"
    job["out_dir"] = f"/vol/frames/{tag}"
    job["result_json"] = f"/vol/frames/{tag}/render_timing.json"
    job["samples"] = samples
    job["engine"] = engine
    job["device"] = "GPU"
    job["exposure"] = -3.0
    job["denoise"] = denoise
    if limit:
        import numpy as np
        npz = np.load(job["poses"])
        sub = {k: npz[k] for k in npz.files}
        sub["poses"] = npz["poses"][:limit]
        for k in ("yaw", "tilt_gt_deg", "lean_gt", "lean_req", "frame_t"):
            if k in sub:
                sub[k] = sub[k][:limit]
        np.savez_compressed(f"/vol/spike/{tag}_poses_{limit}.npz", **sub)
        job["poses"] = f"/vol/spike/{tag}_poses_{limit}.npz"
    jp = Path(f"/tmp/job_{tag}.json")
    jp.write_text(json.dumps(job))

    p = subprocess.run([sys.executable, "-m", "src.sim.blender_render", str(jp)],
                       capture_output=True, text=True, timeout=3300,
                       cwd="/root")
    log = (p.stdout or "") + "\n--stderr--\n" + (p.stderr or "")
    volume.commit()
    out = {"returncode": p.returncode, "log_tail": log[-3000:]}
    for line in log.splitlines():
        if line.startswith("RENDER_RESULT "):
            out["render"] = json.loads(line[len("RENDER_RESULT "):])
    out["n_files"] = len(list(Path(job["out_dir"]).glob("*.png")))
    return out
