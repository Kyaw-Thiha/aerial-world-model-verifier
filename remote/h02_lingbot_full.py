"""Track H2 Part 3: LingBot-World at full precision (bf16), five roll probes.

Weights: ``robbyant/lingbot-world-fast`` (74 GB, Apache-2.0, 4-step distilled) plus
the VAE and T5 encoder from ``robbyant/lingbot-world-base-cam`` (Apache-2.0). The
upstream repository is mounted into the image; the app uses its own ``WanI2VFast``
pipeline and OpenCV c2w poses, exactly as H01 did for the NF4 build.

    set -a; . ./.env; set +a
    pixi run modal run remote/h02_lingbot_full.py::main --mode download
    pixi run modal run remote/h02_lingbot_full.py::main --mode probe
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h01_lingbot_inputs"
REPO = Path("/tmp/lingbot")

vol_w = modal.Volume.from_name("awmv-h2-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-h2-results", create_if_missing=True)
DEST = "/w/lingbot_fast"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "scipy", "pandas", "pillow", "opencv-python-headless",
                 "diffusers>=0.31.0", "transformers==4.51.3", "tokenizers>=0.20.3",
                 "accelerate>=1.1.1", "tqdm", "imageio", "imageio-ffmpeg", "easydict",
                 "ftfy", "safetensors", "huggingface_hub")
    .run_commands(
        "pip install --no-cache-dir "
        "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' "
        "|| echo 'flash-attn wheel unavailable'")
    .add_local_dir(str(REPO), "/lingbot", copy=True,
                   ignore=[".git", "**/.git", "**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(INPUTS), "/inputs", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("awmv-h2-lingbot-full", image=image)


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/w": vol_w})
def download() -> dict:
    import os

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")  # public repos; kept for symmetry
    snapshot_download("robbyant/lingbot-world-base-cam", local_dir=DEST, token=token,
                      allow_patterns=["Wan2.1_VAE.pth",
                                       "models_t5_umt5-xxl-enc-bf16.pth",
                                       "google/umt5-xxl/**"])
    snapshot_download("robbyant/lingbot-world-fast",
                      local_dir=f"{DEST}/lingbot_world_fast", token=token)
    vol_w.commit()
    files = sorted(p for p in Path(DEST).rglob("*") if p.is_file())
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "biggest": sorted(((p.stat().st_size, str(p.relative_to(DEST)))
                               for p in files), reverse=True)[:5]}


def _load_pipeline():
    import os
    import sys

    sys.path.insert(0, "/lingbot")
    import wan
    from wan.configs import WAN_CONFIGS

    # WanI2VFast infers `control_type` from the checkpoint directory name; the
    # fast release has no 'cam' in its name, so expose it through a cam-named
    # symlink (files are unchanged).
    cam_dir = "/w/lingbot_fast_cam"
    if not os.path.exists(cam_dir):
        os.symlink(DEST, cam_dir)
    pipe = wan.WanI2VFast(
        config=WAN_CONFIGS["i2v-A14B"],
        checkpoint_dir=cam_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=False,
    )
    # The upstream wrapper leaves the bundled VAE on CPU in this environment;
    # the pipeline feeds CUDA latents into it. Put it on the GPU explicitly.
    pipe.vae.model.to("cuda")
    return pipe


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=6 * 3600,
              volumes={"/w": vol_w, "/r": vol_r}, env={"HF_HOME": "/w/hf"})
def probe(jobs: str, frame_num: int = 49, max_area: int = 480 * 832) -> dict:
    import os
    import shutil
    import time

    import torch
    from PIL import Image

    payload = json.loads(jobs)
    os.makedirs("/r/lingbot_full", exist_ok=True)
    t0 = time.time()
    pipe = _load_pipeline()
    load_s = time.time() - t0
    out = {"load_s": load_s, "gpu": torch.cuda.get_device_name(0),
           "frames": frame_num, "max_area": max_area, "clips": {}}
    for job in payload:
        jid = job["id"]
        folder = f"/r/lingbot_full/{jid}"
        if os.path.exists(f"{folder}/manifest.json"):
            print("skip", jid)
            continue
        work = f"/tmp/action/{jid}"
        os.makedirs(work, exist_ok=True)
        shutil.copyfile(f"/inputs/{job['pose_dir']}/poses.npy", f"{work}/poses.npy")
        shutil.copyfile(f"/inputs/{job['pose_dir']}/intrinsics.npy",
                        f"{work}/intrinsics.npy")
        img = Image.open(f"/inputs/{job['image']}").convert("RGB")
        # Upstream `WanI2VFast.generate` offloads the transformer at the end but never
        # reloads it for the next call; with several clips in one container every clip
        # after the first would run on CPU. Put the model and VAE back on the GPU.
        pipe.model.to("cuda")
        pipe.vae.model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        video = pipe.generate(
            job["prompt"], img, action_path=work, chunk_size=3, max_area=max_area,
            frame_num=frame_num, shift=5.0, seed=int(job.get("seed", 0)),
            offload_model=True)
        secs = time.time() - t1
        arr = ((video.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        os.makedirs(f"{folder}/frames", exist_ok=True)
        for i, fr in enumerate(arr):
            Image.fromarray(fr).save(f"{folder}/frames/{i:03d}.png")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(f"{folder}/clip.mp4", arr, fps=16)
        except Exception as exc:
            print("mp4 failed", exc, flush=True)
        manifest = {"id": jid, **{k: job[k] for k in job if k != "prompt"},
                    "prompt": job["prompt"], "seconds": secs, "frames": int(len(arr)),
                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu": torch.cuda.get_device_name(0), "precision": "bf16",
                    "repo": "robbyant/lingbot-world-fast"}
        (Path(folder) / "manifest.json").write_text(json.dumps(manifest, indent=1))
        out["clips"][jid] = {"seconds": secs, "vram_peak_gb": manifest["vram_peak_gb"]}
        print(jid, f"{secs:.1f}s", f"{manifest['vram_peak_gb']:.1f}GB", flush=True)
        vol_r.commit()
    out["seconds_total"] = time.time() - t0
    (Path("/r/lingbot_full") / "summary.json").write_text(json.dumps(out, indent=1))
    vol_r.commit()
    return out


@app.local_entrypoint()
def main(mode: str = "download"):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    man = json.loads((INPUTS / "manifest.json").read_text())
    fr = man["frames"][0]
    image = fr["file"].replace("data/h01_lingbot_inputs/", "")
    probe_specs = [("20", "probe_rollramp_p20"), ("-20", "probe_rollramp_m20"),
                   ("fast20", "probe_rollramp_fast20"), ("60", "probe_rollramp_60")]
    level = [g for g in man["grid"] if g["path"] == "cruise" and g["arm"] == "level"
             and g["traj"] == fr["traj"]][0]
    jobs = [{"id": "probe_level", "pose_dir": level["dir"].replace(
                 "data/h01_lingbot_inputs/", ""), "image": image,
             "path": "cruise", "arm": "probe_level", "traj": fr["traj"],
             "frame": fr["frame"],
             "prompt": "A quadrotor drone flies forward over a city, the horizon visible.",
             "seed": 0}]
    for key, jid in probe_specs:
        jobs.append({"id": jid,
                     "pose_dir": man["convention_probe"][key]["dir"].replace(
                         "data/h01_lingbot_inputs/", ""),
                     "image": image, "path": "cruise", "arm": f"probe_roll_{key}",
                     "traj": fr["traj"], "frame": fr["frame"],
                     "prompt": "A quadrotor drone flies forward over a city, the horizon "
                               "visible.", "seed": 0})
    print(f"{len(jobs)} full-precision probe clips")
    res = probe.remote(json.dumps(jobs))
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(json.dumps(res["clips"], indent=1))
