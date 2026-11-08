"""Minimal Modal smoke test: GPU present, Volume read/write, image defined here.

Run once before the first real GPU job:

    pixi run modal run remote/modal_smoke.py

This image is deliberately independent of `pixi.toml` (see docs/decisions.md).
Nothing here is a real workload: it prints `nvidia-smi`, writes a file to a Volume,
reads it back, and exits.
"""
from __future__ import annotations

import modal

app = modal.App("awmv-smoke")

volume = modal.Volume.from_name("awmv-smoke-vol", create_if_missing=True)
VOLUME_MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps")          # provides nvidia-smi if the base image lacks it
    .pip_install("numpy==2.1.3")    # pinned independently of pixi; see decisions.md
)


@app.function(image=image, gpu="T4", volumes={VOLUME_MOUNT: volume}, timeout=300)
def smoke(tag: str = "smoke") -> dict:
    import os
    import subprocess
    import sys

    path = os.path.join(VOLUME_MOUNT, f"{tag}.txt")
    payload = f"{tag}: written by modal smoke test\n"
    with open(path, "w") as fh:
        fh.write(payload)
    volume.commit()

    with open(path) as fh:
        read_back = fh.read()

    nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    out = {
        "python": sys.version.split()[0],
        "volume_path": path,
        "volume_write_read_ok": read_back == payload,
        "gpu_query_rc": nvidia.returncode,
        "nvidia_smi_head": "\n".join(nvidia.stdout.splitlines()[:12]) if nvidia.stdout else nvidia.stderr[:500],
        "torch_cuda": None,
    }
    try:
        import torch  # not installed by this image; reported, not required

        out["torch_cuda"] = bool(torch.cuda.is_available())
        out["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        out["torch_cuda"] = "torch not installed in this image (expected)"
    return out


@app.local_entrypoint()
def main(tag: str = "smoke"):
    import json

    result = smoke.remote(tag)
    print(json.dumps(result, indent=2))
