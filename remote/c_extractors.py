"""Track C: feed-forward pose extractors on Modal — smoke test, cost, and UZH scoring.

Models (all ungated, public):
  da3   Depth Anything 3, DA3-LARGE-1.1 (0.35B, pose + depth, CC BY-NC 4.0 weights)
  vggt  VGGT-1B (pose + depth, CC BY-NC 4.0 weights)
  pi3   Pi3 (pose + pointmaps, BSD-2-Clause weights)

The image is built once (torch 2.5.1 + cu124, the environment MetricTrail documents) and
cached by Modal. Weights cache into the `awmv-c-models` volume under HF_HOME.

Two entry points:
  run_one   cost/behaviour smoke on a 16-frame ANWM clip or 100 TII frames
  run_uzh   score one extractor against Leica truth on a prepared UZH-FPV clip

    pixi run modal run remote/c_extractors.py --model all --source anwm --n-frames 16
    pixi run modal run remote/c_extractors.py --model da3 --source tii --n-frames 100
    pixi run modal run remote/c_extractors.py --uzh --model all --seq indoor_forward_6
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-c-extractors")

data_vol = modal.Volume.from_name("awmv-a-data")     # Track A data, read-only use
c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)   # prepared testbeds
model_vol = modal.Volume.from_name("awmv-c-models", create_if_missing=True)

DA3_ROOT = "/opt/DA3"
VGGT_ROOT = "/opt/vggt"
PI3_ROOT = "/opt/Pi3"
A_DATA = "/a-data"
C_DATA = "/c-data"
MODELS = "/models"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("wheel", "setuptools")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("xformers==0.0.28.post3", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "pillow", "einops", "huggingface_hub", "safetensors", "pandas",
                 "pyyaml", "omegaconf", "typer", "opencv-python-headless", "imageio", "trimesh",
                 "plyfile", "scipy", "matplotlib")
    .run_commands(
        f"git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 {DA3_ROOT}",
        f"git clone --depth 1 https://github.com/facebookresearch/vggt {VGGT_ROOT}",
        f"git clone --depth 1 https://github.com/yyfz/Pi3 {PI3_ROOT}",
        # DA3's api imports every exporter (gsplat, moviepy, colmap) at import time; we only
        # need poses/depth, so stub the exporter package instead of pulling those stacks in
        f"printf '%s\\n' 'def export(*a, **k):' '    raise RuntimeError(\"exporters disabled in Track C image\")' "
        f"> {DA3_ROOT}/src/depth_anything_3/utils/export/__init__.py",
    )
    .env({"HF_HOME": f"{MODELS}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("pycolmap", "moviepy==1.0.3", "imageio-ffmpeg", "open3d", "e3nn", "addict",
                 "evo", "tqdm")
    .add_local_python_source("src")
)


def _infer(model: str, frames, res: int = 0):
    """Run one extractor on RGB uint8 frames; return (c2w [N,4,4], extra, wall_seconds)."""
    import os
    import sys
    import tempfile
    import time

    import numpy as np
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if model == "da3":
        sys.path.insert(0, f"{DA3_ROOT}/src")
        from depth_anything_3.api import DepthAnything3
        net = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to("cuda").eval()
        kw = {"process_res": res} if res else {}
        with torch.no_grad():
            pred = net.inference(frames, **kw)
        wall = time.time() - t0
        ext = np.asarray(pred.extrinsics, dtype=np.float64)       # world-to-camera, [N,3,4]
        n = len(ext)
        c2w = np.tile(np.eye(4), (n, 1, 1))
        c2w[:, :3, :3] = np.transpose(ext[:, :3, :3], (0, 2, 1))
        c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], ext[:, :3, 3])
        depth = getattr(pred, "depth", None)
        extra = {"intrinsics_first": np.asarray(pred.intrinsics)[0].tolist(),
                 "depth_median_first": float(np.median(np.asarray(depth[0]))) if depth is not None else None}

    elif model == "vggt":
        sys.path.insert(0, VGGT_ROOT)
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from PIL import Image
        tmp = tempfile.mkdtemp()
        paths = []
        for i, f in enumerate(frames):
            p = os.path.join(tmp, f"{i:04d}.png")
            Image.fromarray(f).save(p)
            paths.append(p)
        imgs = load_and_preprocess_images(paths).to("cuda")
        net = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tokens, ps_idx = net.aggregator(imgs[None])
            pose_enc = net.camera_head(tokens)[-1]
            ext, intr_t = pose_encoding_to_extri_intri(pose_enc, imgs[None].shape[-2:])
        wall = time.time() - t0
        ext = ext[0].cpu().numpy().astype(np.float64)
        n = len(ext)
        c2w = np.tile(np.eye(4), (n, 1, 1))
        c2w[:, :3, :3] = np.transpose(ext[:, :3, :3], (0, 2, 1))
        c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], ext[:, :3, 3])
        extra = {"intrinsics_first": intr_t[0, 0].cpu().numpy().tolist(),
                 "image_hw_processed": list(imgs.shape[-2:])}

    elif model == "pi3":
        sys.path.insert(0, PI3_ROOT)
        from pi3.models.pi3 import Pi3
        arr = np.stack(frames)
        # Pi3's ViT patch is 14 px: replicate-pad the border up to a multiple
        arr = np.pad(arr, ((0, 0), (0, -arr.shape[1] % 14), (0, -arr.shape[2] % 14), (0, 0)),
                     mode="edge")
        imgs = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        if res:
            imgs = torch.nn.functional.interpolate(imgs, size=(res, res), mode="bilinear",
                                                   align_corners=False)
        imgs = imgs.to("cuda")
        net = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = net(imgs[None])
        wall = time.time() - t0
        c2w = out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        extra = {"image_hw_processed": list(imgs.shape[-2:])}
    else:
        raise ValueError(f"unknown model {model}")

    return c2w, extra, wall


def _anwm_frames(n: int):
    """ANWM Regime A clip A00: 224x224 RGB PNGs, 16 frames, declared 1.5 fps."""
    from pathlib import Path

    import cv2
    d = Path(A_DATA) / "anwm_clips" / "A" / "A00" / "frames"
    paths = sorted(d.glob("*.png"))[:n]
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths], 1.5


def _tii_frames(flight: str, n: int, stride: int = 10, fov: float = 100.0, scale: float = 0.5):
    """TII video -> virtual pinhole (Track A's preparation), 320x240 at stride 10 (~12 Hz)."""
    import sys
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np

    sys.path.insert(0, "/root")
    from src.datasets.tii_video import flight_parts, load_calibration
    from src.perception import undistort as U

    control, track = flight_parts(flight)
    cal = load_calibration(Path(A_DATA) / "tii_video" / "calibration", control, track)
    vc = U.virtual_camera(cal["K"], cal["dist"], (640, 480), fov_deg=fov)
    zp = Path(A_DATA) / "tii_video" / control / f"camera_{flight}.zip"
    out = []
    with zipfile.ZipFile(zp) as z:
        names = sorted(x for x in z.namelist() if x.lower().endswith(".jpg"))[::stride][:n]
        for name in names:
            img = cv2.imdecode(np.frombuffer(z.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
            und = U.undistort(img, vc)
            if scale != 1.0:
                und = cv2.resize(und, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
    return out, 120.0 / stride


@app.function(image=image, gpu="A10G", volumes={A_DATA: data_vol, C_DATA: c_vol, MODELS: model_vol},
              timeout=3600, cpu=4)
def run_one(model: str, source: str, n_frames: int = 16, flight: str = "flight-01p-ellipse",
            stride: int = 10, res: int = 0) -> str:
    """Cost and behaviour of one extractor on one clip (no truth)."""
    import traceback
    from pathlib import Path

    import numpy as np
    import torch

    rec: dict = {"model": model, "source": source, "n_frames": n_frames,
                 "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    try:
        if source == "anwm":
            frames, fps = _anwm_frames(n_frames)
        elif source == "tii":
            frames, fps = _tii_frames(flight, n_frames, stride=stride)
        else:
            raise ValueError(f"unknown source {source}")
        rec.update({"fps": fps, "image_hw": list(frames[0].shape[:2]), "n_loaded": len(frames)})
        c2w, extra, wall = _infer(model, frames, res)
        step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
        rec.update({"wall_s": wall, "s_per_frame": wall / max(len(c2w), 1),
                    "n_poses": int(len(c2w)), "all_finite": bool(np.isfinite(c2w).all()),
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "step_median_units": float(np.median(step)) if len(step) else None,
                    "path_len_units": float(step.sum()) if len(step) else None, **extra})
    except torch.cuda.OutOfMemoryError:
        rec.update({"error": "cuda_oom", "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9})
    except Exception as exc:
        rec.update({"error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1200:]})
    try:
        Path(f"{MODELS}/c-smoke").mkdir(parents=True, exist_ok=True)
        Path(f"{MODELS}/c-smoke/{model}_{source}_{n_frames}.json").write_text(json.dumps(rec, indent=1))
        model_vol.commit()
    except Exception:
        pass
    print(json.dumps(rec, indent=1)[:2000], flush=True)
    return json.dumps(rec)


@app.function(image=image, gpu="A10G", volumes={A_DATA: data_vol, C_DATA: c_vol, MODELS: model_vol},
              timeout=3600, cpu=4)
def run_uzh(model: str, seq: str, max_frames: int = 150, res: int = 0) -> str:
    """Run one extractor on a prepared UZH clip and score it against Leica truth."""
    import traceback
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from src.pathrec.evaluate import score_clip
    from src.pathrec.reliability import jitter_ratio, rotation_rate_deg

    clip_dir = Path(C_DATA) / "uzh" / seq
    truth = np.load(clip_dir / "truth.npz")
    meta = json.loads((clip_dir / "meta.json").read_text())
    frames = []
    for p in sorted((clip_dir / "frames").glob("*.png"))[:max_frames]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))

    rec = {"model": model, "seq": seq, "n_frames": len(frames),
           "gpu": torch.cuda.get_device_name(0), "fps": meta["fps"]}
    try:
        c2w, extra, wall = _infer(model, frames, res)
        n = min(len(c2w), len(truth["t"]))
        q_est = Rotation.from_matrix(c2w[:n, :3, :3]).as_quat()
        sc = score_clip(c2w[:n, :3, 3], truth["pos"][:n], dt=1.0 / float(meta["fps"]),
                        est_quat=q_est, true_quat=truth["quat"][:n], window_s=0.5)
        rec.update({"wall_s": wall, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "n_poses": int(len(c2w)), "all_finite": bool(np.isfinite(c2w).all()),
                    "score": sc,
                    "reliability": {"jitter": jitter_ratio(c2w[:n, :3, 3]),
                                    "rot_rate_deg": rotation_rate_deg(c2w[:n, :3, :3])},
                    **extra})
        out = Path(C_DATA) / "poses" / model / f"{seq}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, c2w=c2w, t=truth["t"][:n])
        c_vol.commit()
    except torch.cuda.OutOfMemoryError:
        rec["error"] = "cuda_oom"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-1200:]
    print(json.dumps(rec, indent=1)[:2500], flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(model: str = "all", source: str = "anwm", n_frames: int = 16, stride: int = 10,
         res: int = 0, uzh: bool = False, seq: str = "", max_frames: int = 150) -> None:
    models = ["da3", "vggt", "pi3"] if model == "all" else [model]
    for m in models:
        try:
            if uzh:
                out = run_uzh.remote(m, seq, max_frames=max_frames, res=res)
                rec = json.loads(out)
                s = rec.get("score") or {}
                print(f"== {m}/{seq}: err={rec.get('error')} "
                      f"dir={s.get('accel_dir_err_deg_median'):.1f}deg "
                      f"ratio={s.get('accel_mag_ratio'):.2f} r2={s.get('lean_r2'):.2f} "
                      f"ate_sim3={s.get('ate_sim3_m'):.2f} ate_se3={s.get('ate_se3_m'):.2f} "
                      f"cov={s.get('coverage_fraction'):.2f} "
                      f"rot={rec.get('score', {}).get('rel_rot_err_deg_median'):.2f}")
            else:
                out = run_one.remote(m, source, n_frames, stride=stride, res=res)
                rec = json.loads(out)
                print(f"== {m}: " + ", ".join(f"{k}={rec.get(k)}" for k in
                      ("error", "wall_s", "s_per_frame", "peak_gpu_gb", "n_poses", "all_finite")
                      if k in rec))
        except Exception as exc:
            print(f"== {m}: FAILED {type(exc).__name__}: {exc}")
