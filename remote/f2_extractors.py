"""Track F2, part 3: feed-forward extractors on the ten UZH sequences.

Two models are runnable from open artefacts; FFVO has no code release at the time of writing
(recorded in the report and decisions log):

  mapanything  Meta's MapAnything, ``facebook/map-anything-apache`` (Apache-2.0 weights,
               ungated), the metric-3D model. Intrinsics are supplied per view.
  pi3x         Pi3X (``yyfz233/Pi3X``, ungated), the multi-modal successor of Pi3, with the
               Kalibr intrinsics supplied as a conditioning input.

VGGT's released API does not accept intrinsics, so its calibrated run is not possible; the
uncalibrated 150-frame number from Track C's bake-off is quoted in the report instead.

    pixi run modal run remote/f2_extractors.py --method map --probe indoor_forward_5
    pixi run modal run remote/f2_extractors.py --method map --seqs indoor_forward_5
    pixi run modal run remote/f2_extractors.py --method pi3
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-f2-extractors")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
f2_vol = modal.Volume.from_name("awmv-f2-data", create_if_missing=True)
m_vol = modal.Volume.from_name("awmv-f2-models", create_if_missing=True)
C_DATA = "/c-data"
F2_DATA = "/f2-data"
MODELS = "/models"

SEQS = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9"]

map_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ffmpeg")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy<2.3", "wheel", "pip>=24.2", "setuptools>=77", "ninja", "packaging")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/map-anything /opt/map-anything",
        "cd /opt/map-anything && pip install --no-build-isolation -e .",
    )
    .env({"HF_HOME": f"{MODELS}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

pi3_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ffmpeg")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "wheel", "pip>=24.2", "setuptools>=77", "ninja",
                 "packaging", "huggingface_hub", "safetensors", "pillow",
                 "opencv-python-headless", "plyfile", "einops", "timm")
    .run_commands(
        "git clone --depth 1 https://github.com/yyfz/Pi3 /opt/Pi3",
        "cd /opt/Pi3 && pip install --no-build-isolation -e .",
    )
    .env({"HF_HOME": f"{MODELS}/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF":
          "expandable_segments:True"})
)


# --------------------------------------------------------------------------- helpers
def _stage(seq: str, limit: int = 150) -> str:
    import shutil
    from pathlib import Path

    src = Path(C_DATA) / "uzh" / seq / "frames"
    dst = Path("/tmp") / seq / "frames"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in sorted(src.glob("*.png"))[:limit]:
        shutil.copy(p, dst / p.name)
    return str(dst)


def _meta(seq: str) -> dict:
    from pathlib import Path

    meta = json.loads((Path(C_DATA) / "uzh" / seq / "meta.json").read_text())
    fps = meta.get("fps_measured", meta.get("fps"))
    K = meta.get("intrinsics_pinhole") or meta.get("intrinsics")
    return {"fps": float(fps), "K": K, "n_frames": int(meta.get("n", 150))}


def _save(poses, seq: str, method: str, extra: dict) -> dict:
    import numpy as np
    from pathlib import Path

    out = Path(F2_DATA) / "poses" / method
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=np.float64)
    np.savez_compressed(out / f"{seq}.npz", c2w=poses)
    f2_vol.commit()
    return {"seq": seq, "method": method, "n_poses": int(len(poses)),
            "finite": bool(np.isfinite(poses).all()),
            "path_len_units": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                   axis=1).sum()) if len(poses) > 1 else 0.0,
            **extra}


# --------------------------------------------------------------------------- MapAnything
@app.function(image=map_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F2_DATA: f2_vol,
                                                         MODELS: m_vol},
              timeout=3600, cpu=8)
def mapanything(seqs: list[str], limit: int = 150, probe: bool = False) -> str:
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
    recs = []
    for seq in seqs:
        frames = _stage(seq, limit)
        meta = _meta(seq)
        K = meta["K"]
        # NOTE: adding the native K here raises an IndexError inside
        # _encode_and_fuse_ray_dirs (mask [1] vs tensor [392, 518, 3]) because load_images
        # has already resized the images to 392x518 and attached its own ray directions.
        # The released API therefore runs this model with estimated intrinsics; recorded in
        # the report as a reproduction finding rather than worked around.
        views = load_images(frames)
        t0 = time.time()
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True,
                                minibatch_size=None, use_amp=True, amp_dtype="bf16",
                                apply_mask=False, mask_edges=False,
                                apply_confidence_mask=False)
        rec = {"seq": seq, "method": "mapanything", "wall_s": round(time.time() - t0, 1),
               "n_views": len(views)}
        if probe:
            p0 = preds[0]
            rec["keys"] = sorted(str(x) for x in p0.keys())
            rec["shapes"] = {str(x): list(np.asarray(p0[x].detach().cpu()).shape)
                             for x in p0.keys() if hasattr(p0[x], "detach")}
        poses = []
        for p in preds:
            cp = p["camera_poses"]
            cp = cp[0] if cp.ndim == 3 else cp
            poses.append(np.asarray(cp.detach().float().cpu()))
        rec.update(_save(np.stack(poses), seq, "mapanything",
                         {"metric_scaling": [float(np.asarray(
                             p["metric_scaling_factor"].detach().float().cpu()).ravel()[0])
                             if "metric_scaling_factor" in p else None for p in preds]}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


# --------------------------------------------------------------------------- Pi3X
@app.function(image=pi3_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F2_DATA: f2_vol,
                                                         MODELS: m_vol},
              timeout=3600, cpu=8)
def pi3(seqs: list[str], limit: int = 150, probe: bool = False) -> str:
    import time

    import numpy as np
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    recs = []
    for seq in seqs:
        frames = _stage(seq, limit)
        meta = _meta(seq)
        K = meta["K"]
        n = int(len(list(__import__("pathlib").Path(frames).glob("*.png"))))
        K_np = np.tile(np.array([[K["fx"], 0.0, K["cx"]], [0.0, K["fy"], K["cy"]],
                                 [0.0, 0.0, 1.0]], dtype=np.float32), (n, 1, 1))
        conditions = {"intrinsics": K_np, "poses": None, "depths": None}
        imgs, cond = load_multimodal_data(frames, conditions, interval=1, device="cuda")
        t0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                res = model(imgs=imgs, **cond)
        rec = {"seq": seq, "method": "pi3x", "wall_s": round(time.time() - t0, 1),
               "n_views": int(imgs.shape[1])}
        if probe:
            rec["keys"] = sorted(str(x) for x in res.keys())
            rec["shapes"] = {str(x): list(np.asarray(res[x].detach().cpu()).shape)
                             for x in res.keys() if hasattr(res[x], "detach")}
        if "camera_poses" in res:
            poses = res["camera_poses"][0].detach().float().cpu().numpy()
        elif "poses" in res:
            poses = res["poses"][0].detach().float().cpu().numpy()
        else:
            rec["error"] = "no camera_poses key"
            recs.append(rec)
            continue
        rec.update(_save(poses, seq, "pi3x", {}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(method: str = "map", seqs: str = "", limit: int = 150, probe: bool = False):
    chosen = [s for s in seqs.split(",") if s] or SEQS
    fn = {"map": mapanything, "pi3": pi3}[method]
    print(fn.remote(chosen, limit, probe))
