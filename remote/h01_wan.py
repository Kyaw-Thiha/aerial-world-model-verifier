"""Track H (H01) Part 3: Wan 2.2 image-to-video, descriptive cost/tilt pilot.

Model: ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` (Apache-2.0), the smaller Wan 2.2 variant. The
A14B I2V diffusers release is ~96 GB of transformer weights and the README's inference path
is the original repo's multi-GPU script, so the 5B TI2V (24 GB VRAM, diffusers-native I2V)
is used; the choice and its consequence are recorded in the report.

    set -a; . ./.env; set +a
    pixi run modal run remote/h01_wan.py::download
    pixi run modal run remote/h01_wan.py::clips
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h01_lingbot_inputs"

vol_w = modal.Volume.from_name("awmv-h-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-h-results", create_if_missing=True)

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEST = "/w/wan22_ti2v5b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 "
        "--index-url https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "scipy", "pillow", "imageio", "imageio-ffmpeg", "ftfy",
                 "accelerate", "transformers==4.51.3", "diffusers==0.35.1",
                 "huggingface_hub", "safetensors")
    .add_local_dir(str(INPUTS), "/inputs", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("awmv-h-wan", image=image)


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/w": vol_w})
def download() -> dict:
    import os

    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, local_dir=DEST, token=os.environ.get("HF_TOKEN"))
    vol_w.commit()
    files = sorted(p for p in Path(DEST).rglob("*") if p.is_file())
    return {"path": path, "files": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


PROMPTS = [
    ("banked_turn_right", "A quadrotor drone banks steeply to the right while turning at "
                          "constant speed over a city."),
    ("slalom", "A drone flies a fast slalom, banking left then right between buildings."),
    ("approach", "A drone approaches a landing pad, pitching nose down and slowing."),
    ("orbit", "A drone performs a constant-speed orbit around a tower, tilted toward the "
              "centre."),
    ("climb_bank_left", "A drone climbs and banks left over an urban skyline."),
]


@app.function(gpu="A100-80GB", timeout=4 * 3600, volumes={"/w": vol_w, "/r": vol_r},
              env={"HF_HOME": "/w/hf"})
def clips(frame_num: int = 49, steps: int = 40, guidance: float = 5.0,
          height: int = 480, width: int = 832, limit: int = 0) -> dict:
    import time

    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    man = json.loads((Path("/inputs") / "manifest.json").read_text())
    frames = man["frames"]
    model_id = DEST
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae,
                                                   torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    out = {"gpu": torch.cuda.get_device_name(0), "frame_num": frame_num, "steps": steps,
           "height": height, "width": width, "clips": {}}
    t0 = time.time()
    jobs = PROMPTS[:limit] if limit else PROMPTS
    for i, (slug, prompt) in enumerate(jobs):
        fr = frames[i % len(frames)]
        folder = Path("/r/wan") / slug
        folder.mkdir(parents=True, exist_ok=True)
        img = Image.open(Path("/inputs") / fr["file"].replace(
            "data/h01_lingbot_inputs/", "")).convert("RGB")
        ts = time.time()
        res = pipe(image=img, prompt=prompt, height=height, width=width,
                   num_frames=frame_num, guidance_scale=guidance,
                   num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(0))
        video = res.frames[0]
        secs = time.time() - ts
        (folder / "frames").mkdir(exist_ok=True)
        for k, arr in enumerate(video):
            arr.save(folder / "frames" / f"{k:03d}.png")
        export_to_video(video, str(folder / "clip.mp4"), fps=24)
        manifest = {"id": slug, "prompt": prompt, "seconds": secs,
                    "frames": len(video), "height": height, "width": width,
                    "steps": steps, "guidance_scale": guidance,
                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu": torch.cuda.get_device_name(0), "first_frame": fr,
                    "model": REPO}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
        out["clips"][slug] = {"seconds": secs, "frames": len(video),
                              "vram_peak_gb": manifest["vram_peak_gb"]}
        print(slug, f"{secs:.1f}s", f"{manifest['vram_peak_gb']:.1f}GB", flush=True)
        vol_r.commit()
    out["seconds_total"] = time.time() - t0
    (Path("/r/wan") / "summary.json").write_text(json.dumps(out, indent=1))
    vol_r.commit()
    return out


@app.local_entrypoint()
def main(mode: str = "download", frame_num: int = 49, steps: int = 40):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    res = clips.remote(frame_num=frame_num, steps=steps)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(json.dumps(res["clips"], indent=1))
