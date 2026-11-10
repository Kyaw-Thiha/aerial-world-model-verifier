"""Track C, Task 1: availability/cost smoke test of feed-forward pose extractors on Modal.

Three extractors share one image (torch 2.5.1 + cu124, matching MetricTrail's documented
environment so the same image can host MetricTrail later):

* ``da3``  Depth Anything 3 (ByteDance-Seed/Depth-Anything-3), DA3-LARGE-1.1, pose + depth,
* ``vggt`` VGGT-1B (facebookresearch/vggt), pose + depth, scale-free,
* ``pi3``  Pi3 (yyfz/Pi3), pose + depth, scale-free.

What is measured per (model, clip): wall time, s/frame, peak GPU memory, pose count, the
fraction of finite poses, and the recovered intrinsics, so the Task 1 cost table has
numbers instead of guesses. Accuracy is *not* judged here (no ground truth in the loop);
that is the Day-2 bake-off.

Frames come from Track A's volume read-only: ANWM clip A/A00 (16 frames, 224x224) and the
TII piloted 01p video (undistorted virtual pinhole, 320x240), the same preparation Track A
fed DPVO.

    pixi run modal run remote/c_extractors.py --model da3 --source anwm --n-frames 16
    pixi run modal run remote/c_extractors.py --model all --source tii --n-frames 100
"""
from __future__ import annotations

import modal

app = modal.App("awmv-c-extractors")

data_vol = modal.Volume.from_name("awmv-a-data")      # read-only use: frames and calibration
model_vol = modal.Volume.from_name("awmv-c-models", create_if_missing=True)

DA3_ROOT = "/opt/DA3"
VGGT_ROOT = "/opt/vggt"
PI3_ROOT = "/opt/Pi3"
A_DATA = "/a-data"
MODELS = "/models"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("wheel", "setuptools")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("xformers==0.0.28.post3", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "pillow", "einops", "huggingface_hub", "safetensors",
                 "omegaconf", "typer", "opencv-python-headless", "imageio", "trimesh",
                 "plyfile", "scipy", "matplotlib")
    .run_commands(
        f"git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 {DA3_ROOT}",
        f"git clone --depth 1 https://github.com/facebookresearch/vggt {VGGT_ROOT}",
        f"git clone --depth 1 https://github.com/yyfz/Pi3 {PI3_ROOT}",
    )
    .env({"HF_HOME": f"{MODELS}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("pycolmap")
    .add_local_python_source("src")
)


def _anwm_frames(n: int):
    """ANWM Regime A clip A00: 224x224 RGB PNGs, 16 frames, ~1.5 fps (declared)."""
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
        names = sorted(n_ for n_ in z.namelist() if n_.lower().endswith(".jpg"))[::stride][:n]
        for name in names:
            img = cv2.imdecode(np.frombuffer(z.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
            und = U.undistort(img, vc)
            if scale != 1.0:
                und = cv2.resize(und, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
    dt = 1.0 / (120.0 / stride)
    return out, 1.0 / dt


@app.function(image=image, gpu="A10G", volumes={A_DATA: data_vol, MODELS: model_vol},
              timeout=3600, cpu=4)
def run_one(model: str, source: str, n_frames: int = 16, flight: str = "flight-01p-ellipse",
            stride: int = 10, res: int = 0) -> dict:
    import json
    import time
    import traceback
    import sys

    import numpy as np
    import torch

    rec: dict = {"model": model, "source": source, "n_frames": n_frames,
                 "gpu": torch.cuda.get_device_name(0),
                 "torch": torch.__version__}
    try:
        if source == "anwm":
            frames, fps = _anwm_frames(n_frames)
        elif source == "tii":
            frames, fps = _tii_frames(flight, n_frames, stride=stride)
        else:
            raise ValueError(f"unknown source {source}")
        rec.update({"fps": fps, "image_hw": list(frames[0].shape[:2]), "n_loaded": len(frames)})

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
            ext = np.asarray(pred.extrinsics, dtype=np.float64)   # w2c, [N,3,4]
            N = len(ext)
            c2w = np.tile(np.eye(4), (N, 1, 1))
            c2w[:, :3, :3] = np.transpose(ext[:, :3, :3], (0, 2, 1))
            c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], ext[:, :3, 3])
            intr = np.asarray(pred.intrinsics, dtype=np.float64)
            depth = getattr(pred, "depth", None)
            extra = {"intrinsics_first": intr[0].tolist() if intr.size else None,
                     "has_metric_depth": depth is not None,
                     "depth_median_first": float(np.median(np.asarray(depth[0]))) if depth is not None else None}

        elif model == "vggt":
            sys.path.insert(0, VGGT_ROOT)
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            import tempfile, os
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
            ext = ext[0].cpu().numpy().astype(np.float64)      # w2c [N,3,4]
            N = len(ext)
            c2w = np.tile(np.eye(4), (N, 1, 1))
            c2w[:, :3, :3] = np.transpose(ext[:, :3, :3], (0, 2, 1))
            c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], ext[:, :3, 3])
            extra = {"intrinsics_first": intr_t[0, 0].cpu().numpy().tolist(),
                     "has_metric_depth": False,
                     "image_hw_processed": list(imgs.shape[-2:])}

        elif model == "pi3":
            sys.path.insert(0, PI3_ROOT)
            from pi3.models.pi3 import Pi3
            net = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
            imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
            if res:
                imgs = torch.nn.functional.interpolate(imgs, size=(res, res), mode="bilinear",
                                                       align_corners=False)
            imgs = imgs.to("cuda")
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = net(imgs[None])
            wall = time.time() - t0
            c2w = out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
            N = len(c2w)
            extra = {"intrinsics_first": None, "has_metric_depth": False,
                     "image_hw_processed": list(imgs.shape[-2:])}
        else:
            raise ValueError(f"unknown model {model}")

        finite = bool(np.isfinite(c2w).all())
        step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
        rec.update({
            "wall_s": wall, "s_per_frame": wall / max(N, 1), "n_poses": int(N),
            "all_finite": finite,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "step_median_units": float(np.median(step)) if len(step) else None,
            "step_p95_units": float(np.percentile(step, 95)) if len(step) else None,
            "path_len_units": float(step.sum()) if len(step) else None,
            **extra,
        })
    except torch.cuda.OutOfMemoryError:
        rec.update({"error": "cuda_oom",
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9})
    except Exception as exc:  # report, do not fail the batch
        rec.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})

    tag = f"{model}_{source}_{n_frames}_{rec.get('image_hw')}"
    try:
        from pathlib import Path
        Path(f"{MODELS}/c-smoke").mkdir(parents=True, exist_ok=True)
        Path(f"{MODELS}/c-smoke/{tag}.json").write_text(json.dumps(rec, indent=1))
        model_vol.commit()
    except Exception:
        pass
    print(json.dumps(rec, indent=1)[:2500], flush=True)
    return rec


@app.local_entrypoint()
def main(model: str = "all", source: str = "anwm", n_frames: int = 16, stride: int = 10,
         res: int = 0) -> None:
    models = ["da3", "vggt", "pi3"] if model == "all" else [model]
    for m in models:
        try:
            rec = run_one.remote(m, source, n_frames, stride=stride, res=res)
            print(f"== {m}: {json_summary(rec)}")
        except Exception as exc:
            print(f"== {m}: FAILED {type(exc).__name__}: {exc}")


def json_summary(rec: dict) -> str:
    keys = ("error", "wall_s", "s_per_frame", "peak_gpu_gb", "n_poses", "all_finite",
            "path_len_units", "image_hw_processed", "has_metric_depth")
    return ", ".join(f"{k}={rec.get(k)}" for k in keys if k in rec)
