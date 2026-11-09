"""Modal image and jobs for ANWM (Track B, prefixes awmv-b-*).

ANWM needs its own Python/torch/diffusers pins, so it never runs in pixi. Volumes:
  awmv-b-weights   /w   slim EMA checkpoint (+ the source file while extracting)
  awmv-b-hf-cache  /hf  diffusers/torch hub cache (sd-vae-ft-ema)
  awmv-b-native    /n   native trajectories pulled from the HF dataset shards
  awmv-b-results   /r   rollout outputs

Usage:
  modal run remote/b_anwm.py::extract_slim
  modal run remote/b_anwm.py::smoke
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP = "awmv-b-anwm"
CKPT_REPO, CKPT_FILE = "EmbodiedCity/ANWM", "0200000.pth.tar"

vol_weights = modal.Volume.from_name("awmv-b-weights", create_if_missing=True)
vol_hf = modal.Volume.from_name("awmv-b-hf-cache", create_if_missing=True)
vol_native = modal.Volume.from_name("awmv-b-native", create_if_missing=True)
vol_results = modal.Volume.from_name("awmv-b-results", create_if_missing=True)

ANWM_LOCAL = Path(__file__).resolve().parents[1] / "third_party" / "anwm"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install --no-cache-dir 'numpy<2'",
    )
    .run_commands(
        "python -c \"import torch;print('torch', torch.__version__, torch.cuda.is_available())\""
    )
    .add_local_dir(str(ANWM_LOCAL), "/anwm", copy=True)
    .run_commands("cd /anwm && pip install --no-cache-dir -e '.[metrics]'",
                  "pip install --no-cache-dir huggingface_hub")
)

app = modal.App(APP, image=image)


# --------------------------------------------------------------------- Task 1.1
@app.function(cpu=8, memory=65536, timeout=3 * 3600, volumes={"/w": vol_weights, "/hf": vol_hf})
def extract_slim() -> dict:
    """One-off: pull the 18.1 GB training checkpoint, keep only the EMA weights (~4.4 GB)."""
    import hashlib

    import torch
    from huggingface_hub import hf_hub_download

    def sha256(path: str, chunk: int = 1 << 25) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()

    os.makedirs("/w/slim", exist_ok=True)
    src = hf_hub_download(CKPT_REPO, CKPT_FILE, cache_dir="/hf/hub")
    src_size = os.path.getsize(src)
    print(f"source {src} {src_size/1e9:.2f} GB")
    # mmap avoids holding 18 GB of tensors resident while copying the EMA
    ckp = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    print("checkpoint keys:", sorted(ckp.keys())[:12], "step:", ckp.get("step"))
    slim = {"ema": {k: v.clone() for k, v in ckp["ema"].items()}, "step": ckp.get("step"),
            "source": f"{CKPT_REPO}/{CKPT_FILE}", "source_bytes": src_size}
    dst = "/w/slim/0200000_ema.pth.tar"
    torch.save(slim, dst)
    del ckp, slim
    info = {"source": src, "source_bytes": src_size, "source_sha256": sha256(src),
            "slim": dst, "slim_bytes": os.path.getsize(dst), "slim_sha256": sha256(dst)}
    Path("/w/slim/manifest.json").write_text(json.dumps(info, indent=1))
    vol_weights.commit()
    print(json.dumps(info, indent=1))
    return info


# --------------------------------------------------------------------- Task 1.3
def _load_models(device: str = "cuda"):
    import torch
    from diffusers.models import AutoencoderKL

    from anwm.diffusion import create_diffusion
    from anwm.model import CDiT_models

    model = CDiT_models["CDiT-XL/2"](context_size=4, input_size=28, in_channels=4)
    ckp = torch.load("/w/slim/0200000_ema.pth.tar", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckp["ema"], strict=True)
    print("load_state_dict:", msg)
    model.eval().to(device)
    diffusion = create_diffusion(str(250))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    return model, diffusion, vae


@app.function(gpu="L4", timeout=1800, volumes={"/w": vol_weights, "/hf": vol_hf},
              env={"HF_HOME": "/hf"})
def smoke(steps: int = 250, reps: int = 3) -> dict:
    """One 224x224-equivalent forward pass and one full 250-step frame, timed and measured."""
    import torch

    from anwm.rollout import model_forward_wrapper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    t0 = time.time()
    model, diffusion, vae = _load_models(device)
    load_s = time.time() - t0
    params = sum(p.numel() for p in model.parameters())

    B, T = 1, 16
    y = torch.randn(B, 4, device=device)
    x_cond = torch.randn(B, 4, 4, 28, 28, device=device)
    rel_t = torch.full((B,), T / 128.0, device=device)
    x_sup = torch.randn(B, 1, 4, 28, 28, device=device)

    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for _ in range(reps):  # warmup + timed passes
            x = torch.randn(B, 4, 28, 28, device=device)
            tt = torch.full((B,), 500, device=device, dtype=torch.long)
            torch.cuda.synchronize()
            s = time.time()
            model.forward(x, tt, y=y, x_cond=x_cond, rel_t=rel_t, x_supervised=x_sup)
            torch.cuda.synchronize()
            times.append(time.time() - s)
    per_pass = min(times)
    peak_forward = torch.cuda.max_memory_allocated() / 1e9

    # one complete generated frame: 250 passes through the sampler + VAE round trip
    torch.cuda.reset_peak_memory_stats()
    obs = torch.randn(B, 4, 3, 224, 224, device=device)
    fsup = torch.randn(B, 1, 3, 224, 224, device=device)
    torch.cuda.synchronize()
    s = time.time()
    out = model_forward_wrapper((model, diffusion, vae), obs, y, 1, 28, device=device,
                                num_cond=4, num_goals=1, x_supervised=fsup)
    torch.cuda.synchronize()
    frame_s = time.time() - s
    peak_frame = torch.cuda.max_memory_allocated() / 1e9
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "params_b": params / 1e9, "load_s": load_s,
        "pass_ms": 1e3 * per_pass, "pass_ms_all_reps": [1e3 * t for t in times],
        "frame_s": frame_s, "frame_shape": list(out.shape),
        "rollout16_est_s": 16 * frame_s,
        "peak_forward_gb": peak_forward, "peak_frame_gb": peak_frame,
        "sampling_steps": steps,
    }
    Path("/w/smoke.json").write_text(json.dumps(info, indent=1))
    vol_weights.commit()
    print(json.dumps(info, indent=1))
    return info
