"""Track H (H01): LingBot-World (community NF4 build) on Modal.

Weights: ``cahlen/lingbot-world-base-cam-nf4`` — a self-contained NF4 quantisation of
LingBot-World-Base-Cam (Apache-2.0), ~33 GB, advertised to fit 32 GB VRAM, camera-pose
control supported. The upstream repository's 8-GPU fast variant is recorded but not used
here; this route avoids flash-attn and runs on one GPU.

    set -a; . ./.env; set +a
    pixi run modal run remote/h01_lingbot.py::download
    pixi run modal run remote/h01_lingbot.py::one --job <grid key>

Inputs are built locally by ``experiments/h01_lingbot_inputs.py`` and mounted read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h01_lingbot_inputs"

vol_w = modal.Volume.from_name("awmv-h-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-h-lingbot-results-b", create_if_missing=True)

REPO = "cahlen/lingbot-world-base-cam-nf4"
DEST = "/w/lingbot_nf4"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 "
        "--index-url https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "scipy", "pandas", "pillow", "opencv-python-headless",
                 "bitsandbytes>=0.43.0", "safetensors", "accelerate", "transformers==4.51.3",
                 "diffusers==0.31.0", "einops", "imageio", "imageio-ffmpeg", "easydict",
                 "ftfy", "tqdm", "huggingface_hub")
    .run_commands("pip install --no-cache-dir "
                  "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
                  "flash_attn-2.6.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' "
                  "|| echo 'flash-attn wheel unavailable; SDPA fallback will be used'")
    .add_local_dir(str(INPUTS), "/inputs", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("awmv-h-lingbot-b", image=image)


@app.function(cpu=16, memory=32768, timeout=3 * 3600, volumes={"/w": vol_w})
def download() -> dict:
    import os

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    path = snapshot_download(REPO, local_dir=DEST, token=token,
                             ignore_patterns=["**/__pycache__/**", "*.pyc"])
    vol_w.commit()
    files = sorted(p for p in Path(DEST).rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    return {"path": path, "files": len(files), "bytes": size,
            "biggest": sorted(((p.stat().st_size, str(p.relative_to(DEST)))
                               for p in files), reverse=True)[:5]}


def _patch_attention():
    """Use PyTorch's SDPA when flash-attn is absent (the model calls ``flash_attention`` directly)."""
    import torch
    import torch.nn.functional as F

    sys_path = "/w/lingbot_nf4"
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from wan.modules import attention as att
    import wan.modules.model as wm

    try:
        import wan.modules.model_fast as wmf
    except Exception:
        wmf = None
    if att.FLASH_ATTN_2_AVAILABLE or att.FLASH_ATTN_3_AVAILABLE:
        return "flash-attn"

    def sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
             q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
             dtype=torch.bfloat16, version=None):
        orig = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            q = q * q_scale
        if k.shape[2] != q.shape[2]:
            rep = q.shape[2] // k.shape[2]
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        mask = None
        if k_lens is not None:
            lens = torch.as_tensor(k_lens, device=k.device).flatten()
            if int(lens.min()) < k.shape[1]:
                ar = torch.arange(k.shape[1], device=k.device)[None, :]
                mask = (ar >= lens[:, None])[:, None, None, :]
                mask = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(
                    mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                             dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().to(orig)

    wm.flash_attention = sdpa
    att.flash_attention = sdpa
    if wmf is not None:
        wmf.flash_attention = sdpa
    return "sdpa"


def _load_pipeline():
    import sys

    sys.path.insert(0, DEST)
    mode = _patch_attention()
    print("attention backend:", mode, flush=True)
    from generate_prequant import WanI2V_PreQuant

    return WanI2V_PreQuant(checkpoint_dir=DEST, device_id=0, t5_cpu=False)


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=6 * 3600,
              volumes={"/w": vol_w, "/r": vol_r},
              env={"HF_HOME": "/w/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def generate(jobs: str, sampling_steps: int = 40, frame_num: int = 49,
             max_area: int = 480 * 832) -> dict:
    """Generate every job in one container and record seconds, peak VRAM and GPU."""
    import os
    import shutil
    import time

    import torch
    from PIL import Image

    payload = json.loads(jobs)
    os.makedirs("/r/lingbot", exist_ok=True)
    t0 = time.time()
    pipe = _load_pipeline()
    load_s = time.time() - t0
    out = {"load_s": load_s, "gpu": torch.cuda.get_device_name(0),
           "sampling_steps": sampling_steps, "frame_num": frame_num,
           "max_area": max_area, "clips": {}}
    for job in payload:
        jid = job["id"]
        folder = f"/r/lingbot/{jid}"
        if os.path.exists(f"{folder}/manifest.json"):
            print("skip", jid)
            continue
        work = f"/tmp/action/{jid}"
        os.makedirs(work, exist_ok=True)
        shutil.copyfile(f"/inputs/{job['pose_dir']}/poses.npy", f"{work}/poses.npy")
        shutil.copyfile(f"/inputs/{job['pose_dir']}/intrinsics.npy", f"{work}/intrinsics.npy")
        img = Image.open(f"/inputs/{job['image']}").convert("RGB")
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        video = pipe.generate(job["prompt"], img, action_path=work, max_area=max_area,
                              frame_num=frame_num, sampling_steps=sampling_steps,
                              guide_scale=5.0, seed=int(job.get("seed", 0)))
        secs = time.time() - t1
        # video: (C, F, H, W) in [-1, 1]
        arr = ((video.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        os.makedirs(f"{folder}/frames", exist_ok=True)
        for i, fr in enumerate(arr):
            Image.fromarray(fr).save(f"{folder}/frames/{i:03d}.png")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(f"{folder}/clip.mp4", arr, fps=16)
        except Exception as exc:  # a missing codec must not lose the frames
            print("mp4 write failed:", exc, flush=True)
        manifest = {"id": jid, **{k: job[k] for k in job if k not in ("prompt",)},
                    "prompt": job["prompt"], "seconds": secs,
                    "seconds_per_frame": secs / len(arr), "frames": int(len(arr)),
                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                    "gpu": torch.cuda.get_device_name(0), "sampling_steps": sampling_steps,
                    "frame_num": frame_num, "max_area": max_area,
                    "hf_repo": REPO}
        (Path(folder) / "manifest.json").write_text(json.dumps(manifest, indent=1))
        out["clips"][jid] = {"seconds": secs, "vram_peak_gb": manifest["vram_peak_gb"],
                             "frames": len(arr)}
        print(jid, f"{secs:.1f}s", f"{secs/len(arr):.2f}s/frame",
              f"{manifest['vram_peak_gb']:.1f}GB", flush=True)
        vol_r.commit()
    out["seconds_total"] = time.time() - t0
    (Path("/r/lingbot") / "summary.json").write_text(json.dumps(out, indent=1))
    vol_r.commit()
    return out


@app.local_entrypoint()
def main(mode: str = "download", grid: str = "", limit: int = 0, steps: int = 40,
         frame_num: int = 49):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    man = json.loads((INPUTS / "manifest.json").read_text())
    if mode == "probe":
        probe = man["convention_probe"]
        fr = man["frames"][0]
        image = fr["file"].replace("data/h01_lingbot_inputs/", "")
        level = [g for g in man["grid"] if g["path"] == "cruise" and g["arm"] == "level"
                 and g["traj"] == fr["traj"]][0]
        jobs = [{"id": "probe_level", "pose_dir": level["dir"].replace(
                     "data/h01_lingbot_inputs/", ""),
                 "image": image, "path": "cruise", "arm": "probe_level",
                 "traj": fr["traj"], "frame": fr["frame"],
                 "prompt": "A quadrotor drone flies forward over a city, the horizon "
                           "visible.", "seed": 0}]
        for key, label in (("20", "probe_rollramp_p20"), ("-20", "probe_rollramp_m20")):
            jobs.append({"id": label,
                         "pose_dir": probe[key]["dir"].replace(
                             "data/h01_lingbot_inputs/", ""),
                         "image": image, "path": "cruise", "arm": f"probe_roll_{key}",
                         "traj": fr["traj"], "frame": fr["frame"],
                         "prompt": "A quadrotor drone flies forward over a city, the "
                                   "horizon visible.",
                         "seed": 0})
        print(f"{len(jobs)} probe jobs")
        res = generate.remote(json.dumps(jobs), sampling_steps=steps, frame_num=frame_num)
        print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
        print(json.dumps(res["clips"], indent=1))
        return
    grid_entries = man["grid"]
    if grid:
        keep = set(grid.split(","))
        grid_entries = [g for g in grid_entries if g["path"] in keep]
    if limit:
        grid_entries = grid_entries[:limit]
    pspec = man["paths"]
    prompts = {
        "cruise": "A quadrotor drone flies straight and level at constant speed over a "
                  "sunlit city, camera gliding forward.",
        "turn5": "A quadrotor drone performs a gentle constant-speed banked turn over a "
                 "city, horizon tilting slightly.",
        "turn10": "A quadrotor drone performs a constant-speed banked turn over a city, "
                  "the horizon clearly tilted.",
        "turn20": "A quadrotor drone performs a steep constant-speed banked turn over a "
                  "city, the horizon strongly tilted.",
        "slalom10": "A quadrotor drone flies a slalom, banking left then right between "
                    "buildings at constant speed.",
    }
    jobs = []
    for g in grid_entries:
        jobs.append({"id": f"{g['traj']}_{g['frame']:03d}__{g['path']}__{g['arm']}",
                     "pose_dir": g["dir"].replace("data/h01_lingbot_inputs/", ""),
                     "image": g["image"], "path": g["path"], "arm": g["arm"],
                     "traj": g["traj"], "frame": g["frame"],
                     "prompt": prompts[g["path"]], "seed": 0})
    print(f"{len(jobs)} jobs")
    res = generate.remote(json.dumps(jobs), sampling_steps=steps, frame_num=frame_num)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(json.dumps(res["clips"], indent=1)[:4000])
