"""Track H (H01): ANWM seed generation on Modal.

Same model, image pins and conventions as Track B's ``remote/b_anwm.py`` (verified in
``reports/b05_pose_injection_stage0.md``); the helpers live in
``src/generation/anwm_remote.py`` so this app does not import another track's entrypoint.
Writes to the Track H volume ``awmv-h-results`` (never to Track B's volume).

    set -a; . ./.env; set +a
    pixi run modal run remote/h01_anwm.py::smoke
    pixi run modal run --detach remote/h01_anwm.py::seeds --specs-path results/h01_seed_specs/specs.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

ROOT_DIR = Path(__file__).resolve().parents[1]
ANWM_LOCAL = ROOT_DIR / "third_party" / "anwm"

APP = "awmv-h-anwm-b"
CKPT = "/w/slim/0200000_ema.pth.tar"

vol_weights = modal.Volume.from_name("awmv-b-weights", create_if_missing=False)
vol_hf = modal.Volume.from_name("awmv-b-hf-cache", create_if_missing=False)
vol_native = modal.Volume.from_name("awmv-b-native", create_if_missing=False)
vol_h = modal.Volume.from_name("awmv-h-anwm-results-b", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install --no-cache-dir 'numpy<2'",
    )
    .add_local_dir(str(ANWM_LOCAL), "/anwm", copy=True,
                   ignore=[".git", "**/.git", "**/__pycache__", "**/*.pyc"])
    .run_commands("cd /anwm && pip install --no-cache-dir -e '.[metrics]'",
                  "pip install --no-cache-dir huggingface_hub",
                  "pip install --no-cache-dir 'diffusers==0.33.1' "
                  "'transformers==4.44.2' 'peft==0.13.2'")
    .add_local_dir(str(ROOT_DIR / "src"), "/app/src", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App(APP, image=image)


def _load_models(device: str = "cuda"):
    import torch
    from diffusers.models import AutoencoderKL

    from anwm.diffusion import create_diffusion
    from anwm.model import CDiT_models

    model = CDiT_models["CDiT-XL/2"](context_size=16, input_size=28, in_channels=4)
    ckp = torch.load(CKPT, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckp["ema"], strict=True)
    print("load_state_dict:", msg)
    model.eval().to(device)
    diffusion = create_diffusion(str(250))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    return model, diffusion, vae


@app.function(gpu="A10G", timeout=1800, volumes={"/w": vol_weights, "/hf": vol_hf},
              env={"HF_HOME": "/hf"})
def smoke() -> dict:
    import torch

    t0 = time.time()
    model, diffusion, vae = _load_models()
    load_s = time.time() - t0
    return {"load_s": load_s, "vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            "device": torch.cuda.get_device_name(0)}


@app.function(gpu="A10G", timeout=10 * 3600,
              volumes={"/w": vol_weights, "/hf": vol_hf, "/n": vol_native, "/r": vol_h},
              env={"HF_HOME": "/hf"})
def seeds(specs_json: str, out_root: str = "/r/clips", sampling_steps: int = 250,
          force: bool = False) -> dict:
    """Roll out level-camera clips for every spec and record per-clip seconds and VRAM."""
    import sys

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, "/app")
    from anwm.rollout import model_forward_wrapper
    from anwm.utils import get_delta_np, normalize_data

    from src.generation.anwm_remote import (CTX, camera_axis_change, context_inputs,
                                            eval_shim, load_traj, pose_from_position_yaw)

    specs = json.loads(specs_json)
    device = "cuda"
    model, diffusion, vae = _load_models(device)
    ds = eval_shim()
    cache: dict = {}
    out: dict = {}
    slim_sha = json.load(open("/w/slim/manifest.json"))["slim_sha256"]
    job_t0 = time.time()
    for sp in specs:
        folder = f"{out_root}/{sp['regime']}/{sp['id']}"
        if not force and os.path.exists(f"{folder}/manifest.json"):
            print("skip (already done)", sp["id"])
            continue
        t0 = time.time()
        print("start", sp["id"], flush=True)
        traj = load_traj(cache, sp["start_traj"])
        ct = int(sp["start_time"])
        R_ec, spread = camera_axis_change(traj)
        pos = np.asarray(sp["positions"], float)
        yaw = np.asarray(sp["yaw"], float)
        n = len(pos)
        ctx_times = list(range(ct - CTX + 1, ct + 1))
        target_poses = pose_from_position_yaw(pos[1:n], yaw[1:n], R_ec)
        obs, sup, prior_coverage = context_inputs(ds, traj, ctx_times, target_poses)
        print("context", sp["id"], f"{time.time() - t0:.1f}s", flush=True)

        act, _ = ds._compute_actions({"point": pos, "yaw": yaw.reshape(-1, 1)}, 0,
                                     np.array([n - 1]))
        act[:, :3] = normalize_data(act[:, :3], ds.ACTION_STATS)
        delta = torch.as_tensor(get_delta_np(act), dtype=torch.float32).unsqueeze(0)

        print("diffusion", sp["id"], flush=True)
        torch.manual_seed(int(sp.get("seed", 0)))
        curr = obs.clone().to(device)
        preds = []
        for i in range(n - 1):
            x_pred = model_forward_wrapper(
                (model, diffusion, vae), curr, delta[:, i:i + 1].to(device), 1, 28,
                device=device, num_cond=CTX, num_goals=1,
                x_supervised=sup[:, i:i + 1].to(device))
            curr = torch.cat((curr, x_pred.unsqueeze(1)), dim=1)[:, 1:]
            preds.append(((x_pred.clamp(-1, 1) + 1) / 2 * 255).byte().cpu())
        print("rollout done", sp["id"], f"{time.time() - t0:.1f}s", flush=True)
        os.makedirs(f"{folder}/frames", exist_ok=True)
        for i, t in enumerate(preds):
            Image.fromarray(t[0].permute(1, 2, 0).numpy()).save(f"{folder}/frames/{i:02d}.png")
        secs = time.time() - t0
        stats = {"frame_std": [float(v.float().std() / 255) for v in preds],
                 "temporal_delta": float(np.mean(
                     [float((preds[i].float() - preds[i - 1].float()).abs().mean() / 255)
                      for i in range(1, len(preds))]))}
        manifest = {
            "id": sp["id"], "regime": sp["regime"],
            "start_state": {"traj": sp["start_traj"], "time": ct},
            "commanded_path": {"positions": pos.tolist(), "yaw": yaw.tolist(),
                               "step_spacing_m": float(np.mean(
                                   np.linalg.norm(np.diff(pos, axis=0), axis=1)))},
            "designed_required_tilt_deg": sp.get("designed_tilt_deg"),
            "seed": int(sp.get("seed", 0)), "sampling_steps": sampling_steps,
            "slim_checkpoint_sha256": slim_sha, "arm": "level",
            "camera_axis_change_spread": spread, "meta": sp.get("meta", {}),
            "frames": len(preds), "prior_coverage_per_step": prior_coverage,
            "prior_coverage_mean": float(np.mean(prior_coverage)),
            "prior_coverage_min": float(np.min(prior_coverage)),
            "visual_integrity": stats, "seconds": secs,
            "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu": torch.cuda.get_device_name(0),
        }
        Path(f"{folder}/manifest.json").write_text(json.dumps(manifest, indent=1))
        out[sp["id"]] = {k: manifest[k] for k in
                         ("frames", "seconds", "vram_peak_gb", "gpu", "seed")}
        out[sp["id"]]["visual_integrity"] = stats
        print(sp["id"], "frames", len(preds), f"{secs:.1f}s", f"{secs/len(preds):.1f}s/frame",
              "coverage", round(float(np.mean(prior_coverage)), 3))
        vol_h.commit()
    summary = {"n_clips": len(out), "seconds_total": time.time() - job_t0,
               "seconds_per_clip": (float(np.median([v["seconds"] for v in out.values()]))
                                    if out else None),
               "clips": out}
    Path(f"{out_root}/summary_seedgen.json").write_text(json.dumps(summary, indent=1))
    vol_h.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "smoke", specs_path: str = "results/h01_seed_specs/specs.json",
         out_root: str = "/r/clips"):
    if mode == "smoke":
        print(json.dumps(smoke.remote(), indent=1))
        return
    commit = os.popen("git rev-parse HEAD").read().strip()
    specs = json.load(open(specs_path))
    for s in specs:
        s["code_commit"] = commit
    res = seeds.remote(json.dumps(specs), out_root=out_root)
    print(json.dumps({k: res[k] for k in res if k != "clips"}, indent=1))
