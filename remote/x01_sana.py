"""X01: SANA-WM (bidirectional, Stage 1) camera-trajectory generation on Modal.

Model: ``Efficient-Large-Model/SANA-WM_bidirectional`` (Apache-2.0), the 2.6 B Sana DiT and the
bundled LTX-2 VAE.  The 41 GB LTX-2 refiner and the 46 GB Gemma refiner encoder are **not**
downloaded: ``--no_refiner`` decodes Stage-1 latents with the Sana VAE, and the Sana text
encoder (``Efficient-Large-Model/gemma-2-2b-it``, the release's public mirror) is fetched once
into the weights volume.

Code: ``NVlabs/Sana`` at the commit below, pinned.  The release's own inference script is run
as a subprocess, one call per clip, so the app cannot silently diverge from the release.

    set -a; . ./.env; set +a
    pixi run modal run remote/x01_sana.py::download
    pixi run modal run --detach remote/x01_sana.py::probe
    pixi run modal run --detach remote/x01_sana.py::generate --tag full
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/x01/sana"
IN_CONTAINER = Path("/inputs")
SANA_COMMIT = "3c6650a6f47d41de2fd50a216cd138951d7cbafa"

vol_w = modal.Volume.from_name("awmv-x-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-x-results", create_if_missing=True)
DEST = "/w/sana"
GEMMA = "/w/gemma"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg", "ninja-build", "build-essential",
                 "cmake", "pkg-config")
    .run_commands(
        "pip install -U pip wheel 'setuptools<80'",
        "pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.9.1 torchvision==0.24.1",
        f"git clone https://github.com/NVlabs/Sana.git /sana && cd /sana && "
        f"git checkout {SANA_COMMIT}",
        # mmcv 1.7.2 is imported at import time by diffusion.utils.logger; build it against the
        # torch already installed, exactly as the release's environment_setup.sh does.
        "pip install --no-build-isolation mmcv==1.7.2",
        "cd /sana && pip install --no-cache-dir --no-deps -e .",
        "pip install --no-cache-dir 'numpy<2' pillow scipy opencv-python-headless einops ftfy "
        "regex omegaconf pyyaml tqdm 'imageio[ffmpeg]' pyrallis 'accelerate>=1.3' "
        "transformers==4.57.3 'diffusers>=0.37.0' timm==0.6.13 safetensors "
        "huggingface_hub==0.36.0 sentencepiece protobuf termcolor matplotlib "
        "'flash-linear-attention>=0.4.2' liger_kernel pytz psutil absl-py ml_collections lmdb "
        "webdataset termcolor fire imagesize open_clip_torch easydict qwen-vl-utils",
        "cd /sana && python -c \"import torch, mmcv, diffusers, transformers; "
        "print('torch', torch.__version__, 'mmcv', mmcv.__version__)\"",
    )
    .add_local_dir(str(INPUTS), "/inputs", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("awmv-x-sana", image=image)


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/w": vol_w},
              env={"HF_TOKEN": os.environ.get("HF_TOKEN", ""), "HF_HOME": "/w/hf"})
def download() -> dict:
    import time

    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]
    t0 = time.time()
    snapshot_download("Efficient-Large-Model/SANA-WM_bidirectional", local_dir=DEST,
                      token=token,
                      allow_patterns=["config.yaml", "dit/*", "vae/*"])
    snapshot_download("Efficient-Large-Model/gemma-2-2b-it", local_dir=GEMMA, token=token)
    vol_w.commit()
    files = sorted(p for p in Path("/w").rglob("*") if p.is_file())
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "seconds": time.time() - t0,
            "biggest": sorted(((p.stat().st_size, str(p.relative_to("/w")))
                               for p in files), reverse=True)[:6]}


def _clips(ids: list[str] | None, shard: int, shards: int) -> list[Path]:
    dirs = sorted(p for p in IN_CONTAINER.iterdir() if (p / "camera.npy").exists())
    if ids:
        keep = set(ids)
        dirs = [p for p in dirs if p.name.split("__")[0] in keep or p.name in keep]
    return dirs[shard::shards]


def _generate(ids: list[str] | None, tag: str, shard: int, shards: int, step: int,
              num_frames: int, seed: int) -> dict:
    import subprocess
    import time

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    os.environ["HF_HOME"] = "/w/hf"
    out_root = Path("/r/x01/sana") / tag
    out_root.mkdir(parents=True, exist_ok=True)
    clips = {}
    job_t0 = time.time()
    for d in _clips(ids, shard, shards):
        sid = d.name
        dst = out_root / sid
        if (dst / "manifest.json").exists():
            clips[sid] = json.loads((dst / "manifest.json").read_text())
            continue
        dst.mkdir(parents=True, exist_ok=True)
        work = Path("/tmp") / f"sana_{tag}_{sid}"
        t0 = time.time()
        cmd = [
            "python", "/sana/inference_video_scripts/wm/inference_sana_wm.py",
            "--image", str(d / "image.png"),
            "--prompt", str(d / "prompt.txt"),
            "--camera", str(d / "camera.npy"),
            "--intrinsics", str(d / "intrinsics.npy"),
            "--num_frames", str(num_frames),
            "--fps", "16",
            "--step", str(step),
            "--seed", str(seed),
            "--no_refiner",
            "--offload_vae",
            "--no_action_overlay",
            "--config", f"{DEST}/config.yaml",
            "--model_path", f"{DEST}/dit/sana_wm_1600m_720p.safetensors",
            "--output_dir", str(work),
            "--name", sid,
        ]
        env = dict(os.environ, PYTHONPATH="/sana", TOKENIZERS_PARALLELISM="false")
        proc = subprocess.run(cmd, cwd="/sana", env=env, capture_output=True, text=True)
        (dst / "infer.log").write_text((proc.stdout or "")[-200000:]
                                       + "\n---STDERR---\n" + (proc.stderr or "")[-100000:])
        mp4 = work / f"{sid}_generated.mp4"
        if not mp4.exists():
            cands = sorted(work.glob("*.mp4"))
            mp4 = cands[0] if cands else mp4
        if proc.returncode != 0 or not mp4.exists():
            clips[sid] = {"ok": False, "returncode": proc.returncode,
                          "seconds": time.time() - t0,
                          "tail": ((proc.stderr or proc.stdout or "")[-2000:])}
            (dst / "manifest.json").write_text(json.dumps(clips[sid], indent=1))
            vol_r.commit()
            continue
        import shutil
        shutil.copyfile(mp4, dst / "clip.mp4")
        frames = dst / "frames"
        frames.mkdir(exist_ok=True)
        n = 0
        reader = imageio.get_reader(mp4)
        try:
            for i, fr in enumerate(reader):
                Image.fromarray(np.asarray(fr)).save(frames / f"{i:03d}.png")
                n += 1
        finally:
            reader.close()
        secs = time.time() - t0
        meta = json.loads((d / "meta.json").read_text())
        clips[sid] = {"ok": True, "frames": n, "seconds": secs, "step": step,
                      "num_frames": num_frames, "seed": seed, "path": meta.get("path"),
                      "family": meta.get("family"),
                      "in_envelope": meta.get("in_envelope"), "mp4_bytes": mp4.stat().st_size}
        (dst / "manifest.json").write_text(json.dumps(clips[sid], indent=1))
        vol_r.commit()
        print(sid, n, f"{secs:.1f}s", flush=True)
    summary = {"tag": tag, "shard": shard, "shards": shards, "step": step,
               "num_frames": num_frames, "n_clips": len(clips),
               "n_ok": sum(1 for c in clips.values() if c.get("ok")),
               "seconds_total": time.time() - job_t0, "clips": clips}
    (out_root / f"summary_s{shard}.json").write_text(json.dumps(summary, indent=1))
    vol_r.commit()
    return summary


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=8 * 3600,
              volumes={"/w": vol_w, "/r": vol_r},
              env={"HF_TOKEN": os.environ.get("HF_TOKEN", ""), "HF_HOME": "/w/hf"})
def generate(ids: str = "", tag: str = "full", shard: int = 0, shards: int = 1,
             step: int = 20, num_frames: int = 81, seed: int = 42) -> dict:
    want = [s for s in ids.split(",") if s] if ids else None
    return _generate(want, tag, shard, shards, step, num_frames, seed)


@app.local_entrypoint()
def main(mode: str = "download", ids: str = "", tag: str = "probe", shard: int = 0,
         shards: int = 1, step: int = 20, num_frames: int = 81, seed: int = 42):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    if mode == "probe":
        n = sum(1 for p in INPUTS.iterdir() if (p / "camera.npy").exists())
        print(f"{n} samples available; launching probe on arc_20__S1 and hover_00__S1")
        res = generate.remote(ids="arc_20,hover_00", tag=tag, shard=shard, shards=shards,
                              step=step, num_frames=num_frames, seed=seed)
        print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
        print(json.dumps(res["clips"], indent=1))
        return
    res = generate.remote(ids=ids, tag=tag, shard=shard, shards=shards, step=step,
                          num_frames=num_frames, seed=seed)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
