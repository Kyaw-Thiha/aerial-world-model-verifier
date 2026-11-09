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
    .add_local_dir(str(ANWM_LOCAL), "/anwm", copy=True,
                   ignore=[".git", "**/.git", "**/__pycache__", "**/*.pyc"])
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


# --------------------------------------------------------------------- shared helpers
def _eval_shim(data_folder: str = "/n/airvln_16"):
    """An EvalDataset with attributes set directly, so every method stays ANWM's own code."""
    import numpy as np

    from anwm.data.airvln import EvalDataset
    from anwm.utils import transform

    from anwm.config import load_yaml

    cfg = load_yaml("config/data_config.yaml")
    ds = object.__new__(EvalDataset)
    ds.data_folder = data_folder
    ds.data_split_folder = "/anwm/data/splits/airvln_16"
    ds.dataset_name = "airvln_16"
    ds.image_size = (224, 224)
    ds.min_dist_cat, ds.max_dist_cat = -64, 64
    ds.len_traj_pred, ds.traj_stride, ds.context_size = 16, 80, 4
    ds.normalize = True
    ds.transform = transform
    ds.goals_per_obs = 1
    ds.data_config = cfg["airvln_16"]
    ds.ACTION_STATS = {k: np.expand_dims(np.asarray(v, dtype=np.float32), 0)
                       for k, v in cfg["action_stats"].items()}
    return ds


def _camera_axis_change(traj) -> "np.ndarray":
    """R_ec: the world->camera axis change baked into every released pose.

    The released pose is `[R_z(yaw) @ R_ec | xyz]` (preprocessing notebook, cell 14). Recovering
    R_ec from the data and checking it is constant is the only way to build *new* target poses
    in the same convention; a wrong convention would silently rotate the projected prior.
    """
    import numpy as np

    yaw = np.asarray(traj["yaw"], float).ravel()
    R = np.asarray(traj["pose"], float)[:, :3, :3]
    out = []
    for k in range(len(yaw)):
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        out.append(Rz.T @ R[k])
    out = np.asarray(out)
    spread = float(np.abs(out - out[0]).max())
    if spread > 1e-3:
        raise RuntimeError(f"camera axis change is not constant (spread {spread:.2e})")
    return out[0], spread


def _pose_from_position_yaw(pos, yaw, R_ec):
    import numpy as np

    poses = np.tile(np.eye(4), (len(pos), 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros((len(pos), 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
    Rz[:, 1, 0], Rz[:, 1, 1] = s, c
    Rz[:, 2, 2] = 1.0
    poses[:, :3, :3] = Rz @ R_ec[None]
    poses[:, :3, 3] = pos
    return poses


def _context_inputs(ds, traj, ctx_times, target_poses, data_dir="/n/airvln_16"):
    """(obs_image tensor, x_supervised tensor) through ANWM's own FFP code."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    rgb = []
    for t in ctx_times:
        img = cv2.imread(f"{data_dir}/{traj['name']}/{int(t)}.jpg")
        rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgb = np.stack(rgb, 0)
    proj = ds._compute_projected_images(traj, list(ctx_times), rgb,
                                        np.asarray(target_poses))
    obs = torch.stack([ds.transform(Image.fromarray(rgb[i])) for i in range(len(ctx_times))])
    sup = torch.stack([ds.transform(Image.fromarray(proj[i])) for i in range(len(proj))])
    return obs.unsqueeze(0), sup.unsqueeze(0)


# --------------------------------------------------------------------- Task 2
@app.function(gpu="L4", timeout=3 * 3600, volumes={"/w": vol_weights, "/hf": vol_hf,
                                                    "/n": vol_native, "/r": vol_results},
              env={"HF_HOME": "/hf"})
def validate_native(names: list[str], sampling_steps: int = 250, lpips_net: str = "alex") -> dict:
    """Roll out the released model on native context and compare with the recorded frames."""
    import pickle

    import numpy as np
    import torch
    from PIL import Image

    from anwm.rollout import model_forward_wrapper

    device = "cuda"
    model, diffusion, vae = _load_models(device)
    ds = _eval_shim()
    manifest = json.load(open("/n/manifest.json"))
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net=lpips_net).to(device).eval()

    out = {}
    for name in names:
        ct = manifest[name]["curr_time"]
        if not Path(f"/n/airvln_16/{name}/{ct}.jpg").exists():
            ct = manifest[name]["start_candidates"][0]
            print("fallback start time", name, ct)
        traj = pickle.load(open(f"/n/airvln_16/{name}/traj_data.pkl", "rb"))
        traj = {k: v.astype("float") if hasattr(v, "astype") else v for k, v in traj.items()}
        ds.index_to_data = [(name, ct, -64, 64)]
        idxs, obs, gt, delta, proj = ds[0]
        delta = delta.unsqueeze(0)
        obs, gt, proj = obs.unsqueeze(0), gt.unsqueeze(0), proj.unsqueeze(0)

        preds = []
        curr = obs.clone().to(device)
        for i in range(gt.shape[1]):
            x_pred = model_forward_wrapper(
                (model, diffusion, vae), curr, delta[:, i:i + 1].to(device), 1, 28,
                device=device, num_cond=4, num_goals=1, x_supervised=proj[:, i:i + 1].to(device))
            curr = torch.cat((curr, x_pred.unsqueeze(1)), dim=1)[:, 1:]
            preds.append(x_pred.clamp(-1, 1))
        pred = torch.stack(preds, 1)

        g01 = ((gt.to(device) + 1) / 2).clamp(0, 1)
        p01 = ((pred + 1) / 2).clamp(0, 1)
        d_lpips = [float(lp(2 * p01[:, i] - 1, 2 * g01[:, i] - 1).mean()) for i in range(gt.shape[1])]
        flat = lambda t: t.flatten(1)
        collapse = {
            "pred_frame_std": [float(p01[:, i].std()) for i in range(gt.shape[1])],
            "gt_frame_std": [float(g01[:, i].std()) for i in range(gt.shape[1])],
            "pred_temporal_delta": float((p01[:, 1:] - p01[:, :-1]).abs().mean()),
            "gt_temporal_delta": float((g01[:, 1:] - g01[:, :-1]).abs().mean()),
        }
        folder = f"/r/native/{name}"
        os.makedirs(folder, exist_ok=True)
        tiles = []
        for i in range(gt.shape[1]):
            for tag, t in (("gt", g01), ("pred", p01)):
                Image.fromarray((t[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                ).save(f"{folder}/{i:02d}_{tag}.png")
        # contact sheet: 2 rows (gt, pred) x 16 frames
        rows = []
        for t in (g01, p01):
            ims = [Image.fromarray((t[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                   for i in range(gt.shape[1])]
            w, h = ims[0].size
            row = Image.new("RGB", (w * len(ims), h))
            for i, im in enumerate(ims):
                row.paste(im, (i * w, 0))
            rows.append(row)
        sheet = Image.new("RGB", (rows[0].width, rows[0].height * 2))
        for r, im in enumerate(rows):
            sheet.paste(im, (0, r * rows[0].height))
        sheet.save(f"{folder}/contact_sheet.png")
        out[name] = {"curr_time": int(ct), "n_frames": int(gt.shape[1]), "lpips": d_lpips,
                     "lpips_mean": float(np.mean(d_lpips)), **collapse}
        print(name, "lpips", np.round(d_lpips, 3).tolist())
        vol_results.commit()
    Path("/r/native/summary.json").write_text(json.dumps(out, indent=1))
    vol_results.commit()
    print(json.dumps(out, indent=1)[:2000])
    return out


# --------------------------------------------------------------------- Task 5
@app.function(gpu="L4", timeout=4 * 3600, volumes={"/w": vol_weights, "/hf": vol_hf,
                                                    "/n": vol_native, "/r": vol_results},
              env={"HF_HOME": "/hf"})
def generate(specs_json: str, sampling_steps: int = 250, sync_r2_prefix: str | None = None) -> dict:
    """Roll out the released model from native start states along commanded paths.

    Each spec: {id, regime, positions (17,3), yaw (17,), start_traj, start_time, seed,
    designed_tilt_deg, meta}. Frames go to /r/clips/<regime>/<id>/frames/ + manifest.json.
    """
    import pickle

    import numpy as np
    import torch
    from PIL import Image

    from anwm.rollout import model_forward_wrapper

    specs = json.loads(specs_json)
    device = "cuda"
    model, diffusion, vae = _load_models(device)
    ds = _eval_shim()
    cache = {}
    out = {}

    for sp in specs:
        name, ct = sp["start_traj"], int(sp["start_time"])
        if name not in cache:
            traj = pickle.load(open(f"/n/airvln_16/{name}/traj_data.pkl", "rb"))
            traj = {k: (v.astype("float") if hasattr(v, "astype") else v) for k, v in traj.items()}
            traj["name"] = name
            cache[name] = traj
        traj = cache[name]
        R_ec, spread = _camera_axis_change(traj)

        pos = np.asarray(sp["positions"], float)
        yaw = np.asarray(sp["yaw"], float)
        n = len(pos)
        ctx_times = list(range(ct - 3, ct + 1))
        target_poses = _pose_from_position_yaw(pos[1:n], yaw[1:n], R_ec)
        obs, sup = _context_inputs(ds, traj, ctx_times, target_poses)

        # actions along the commanded path, straight through ANWM's own construction
        class _Shim:
            len_traj_pred = 16
            normalize = True
            data_config = ds.data_config

        act, _ = ds._compute_actions({"point": pos, "yaw": yaw.reshape(-1, 1)}, 0,
                                     np.array([n - 1]))
        from anwm.utils import get_delta_np, normalize_data

        act[:, :3] = normalize_data(act[:, :3], ds.ACTION_STATS)
        delta = torch.as_tensor(get_delta_np(act), dtype=torch.float32).unsqueeze(0)

        torch.manual_seed(int(sp.get("seed", 0)))
        curr = obs.clone().to(device)
        preds = []
        for i in range(n - 1):
            x_pred = model_forward_wrapper(
                (model, diffusion, vae), curr, delta[:, i:i + 1].to(device), 1, 28,
                device=device, num_cond=4, num_goals=1, x_supervised=sup[:, i:i + 1].to(device))
            curr = torch.cat((curr, x_pred.unsqueeze(1)), dim=1)[:, 1:]
            preds.append(((x_pred.clamp(-1, 1) + 1) / 2 * 255).byte().cpu())
        folder = f"/r/clips/{sp['regime']}/{sp['id']}"
        os.makedirs(f"{folder}/frames", exist_ok=True)
        for i, t in enumerate(preds):
            Image.fromarray(t[0].permute(1, 2, 0).numpy()).save(f"{folder}/frames/{i:02d}.png")
        manifest = {
            "id": sp["id"], "regime": sp["regime"], "start_state": {"traj": name, "time": ct},
            "commanded_path": {"positions": pos.tolist(), "yaw": yaw.tolist(),
                              "step_spacing_m": float(np.mean(np.linalg.norm(np.diff(pos, axis=0), axis=1)))},
            "designed_required_tilt_deg": sp.get("designed_tilt_deg"),
            "seed": int(sp.get("seed", 0)), "sampling_steps": sampling_steps,
            "camera_axis_change_spread": spread, "meta": sp.get("meta", {}),
            "frames": len(preds),
        }
        Path(f"{folder}/manifest.json").write_text(json.dumps(manifest, indent=1))
        stats = {"frame_std": [float(v.float().std() / 255) for v in preds],
                 "temporal_delta": float(np.mean([float((preds[i].float() - preds[i - 1].float()).abs().mean() / 255) for i in range(1, len(preds))]))}
        manifest["visual_integrity"] = stats
        Path(f"{folder}/manifest.json").write_text(json.dumps(manifest, indent=1))
        out[sp["id"]] = manifest
        print(sp["id"], "frames", len(preds), "std", round(np.mean(stats["frame_std"]), 4),
              "dt_delta", round(stats["temporal_delta"], 4))
        vol_results.commit()
    Path("/r/clips/summary.json").write_text(json.dumps(out, indent=1))
    vol_results.commit()
    return {k: {"visual_integrity": v["visual_integrity"], "frames": v["frames"]} for k, v in out.items()}


# --------------------------------------------------------------------- Task 4
@app.function(cpu=8, memory=32768, timeout=1800, volumes={"/n": vol_native})
def list_starts(min_forward_m: float = 25.0, min_valid_frac: float = 0.5) -> dict:
    """Which native start states have enough forward depth for the projected prior to be non-empty.

    FFP reprojects the 4 context frames to the target poses, so no future depth is needed; the
    check is that the *scene* ahead of the start has depth out to the path extent. We measure the
    forward depth (90th percentile of valid depth in the central 60% column band) and require it
    to exceed `min_forward_m`, the largest commanded displacement over 16 steps at 5 m/s.
    """
    import pickle

    import numpy as np

    man = json.load(open("/n/manifest.json"))
    out = {}
    for name, m in man.items():
        traj = pickle.load(open(f"/n/airvln_16/{name}/traj_data.pkl", "rb"))
        traj = {k: (v.astype("float") if hasattr(v, "astype") else v) for k, v in traj.items()}
        depth = np.asarray(traj["depth"], float)
        H, W = depth.shape[-2:]
        band = depth[:, :, int(W * 0.2):int(W * 0.8)]
        rows = []
        for ct in m["start_candidates"]:
            d = float(np.percentile(band[ct][band[ct] > 0], 90)) if (band[ct] > 0).any() else 0.0
            valid = float(np.mean(band[ct] > 0))
            path = np.asarray(traj["point"], float)[ct:ct + 17]
            extent = float(np.max(np.linalg.norm(path - path[0], axis=1))) if len(path) > 1 else 0.0
            rows.append({"ct": int(ct), "forward_depth_p90_m": d, "valid_frac": valid,
                         "recorded_path_extent_m": extent,
                         "usable": bool(d > 0 and valid >= min_valid_frac and d > extent),
                         # pose keys so clip specs can be built locally without the pickle
                         "positions": path.tolist() if len(path) == 17 else None,
                         "yaw": np.asarray(traj["yaw"], float)[ct:ct + 17].tolist()})
        out[name] = {"n_frames": int(len(depth)), "candidates": rows}
    Path("/n/starts.json").write_text(json.dumps(out, indent=1))
    vol_native.commit()
    total = sum(r["usable"] for v in out.values() for r in v["candidates"])
    print(json.dumps({k: v["candidates"] for k, v in out.items()}, indent=1)[:3000])
    print("usable starts:", total)
    return out
