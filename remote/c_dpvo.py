"""Track C: DPVO on the UZH testbed (the cell Gate 1 was missing).

Same DPVO build and settings Track A used for TII (commit `0ac95b6`, 48 patches, 0.5x
virtual view), but fed the *prepared* UZH pinhole frames with the real Kalibr intrinsics,
and scored with the same `score_clip` as DA3/VGGT/Pi3.

The image is rebuilt here (it differs from Track A's only by the newer `src` tree), so the
first run pays a ~10 minute CUDA build once and Modal caches it.

    pixi run modal run remote/c_dpvo.py --seq indoor_forward_6
"""
from __future__ import annotations

import modal

app = modal.App("awmv-c-dpvo")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
a_vol = modal.Volume.from_name("awmv-a-data")

C_DATA = "/c-data"
DPVO_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
DPVO_ROOT = "/root/DPVO"
EIGEN_ZIP = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
MODELS_ZIP = ("https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/"
              "models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "unzip", "build-essential", "clang", "libgl1", "libglib2.0-0")
    .pip_install("wheel", "setuptools")
    .pip_install("torch==2.3.0", "torchvision==0.18.0",
                 index_url="https://download.pytorch.org/whl/cu121")
    .run_commands("pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html")
    .pip_install("numpy==1.26.4", "pandas", "pyyaml", "opencv-python-headless", "einops",
                 "pypose", "kornia", "plyfile", "yacs", "evo", "tqdm", "numba", "matplotlib",
                 "scipy")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6"})
    .run_commands(
        f"git clone https://github.com/princeton-vl/DPVO {DPVO_ROOT}",
        f"cd {DPVO_ROOT} && git checkout {DPVO_COMMIT}",
        f"cd {DPVO_ROOT} && wget -q {EIGEN_ZIP} -O eigen.zip && unzip -q eigen.zip -d thirdparty",
        f"cd {DPVO_ROOT} && pip install --no-build-isolation .",
    )
    .add_local_python_source("src")
)


@app.function(image=image, gpu="A10G", volumes={C_DATA: c_vol, "/a-data": a_vol},
              timeout=3600, cpu=2)
def run_uzh_dpvo(seq: str, max_frames: int = 150, patches: int = 48) -> str:
    import json
    import sys
    import time
    import traceback
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, DPVO_ROOT)
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    from src.pathrec.evaluate import score_clip

    clip_dir = Path(C_DATA) / "uzh" / seq
    meta = json.loads((clip_dir / "meta.json").read_text())
    truth = np.load(clip_dir / "truth.npz")
    paths = sorted((clip_dir / "frames").glob("*.png"))[:max_frames]
    frames = [cv2.imread(str(p)) for p in paths]

    rec = {"model": "dpvo", "seq": seq, "n_frames": len(frames),
           "gpu": torch.cuda.get_device_name(0), "dpvo_commit": DPVO_COMMIT}
    try:
        weights = Path("/a-data") / "dpvo_models" / "dpvo.pth"
        cfg.merge_from_file(f"{DPVO_ROOT}/config/default.yaml")
        cfg.PATCHES_PER_FRAME = patches
        cfg.REMOVAL_WINDOW = 12
        cfg.OPTIMIZATION_WINDOW = 8
        cfg.PATCH_LIFETIME = 5

        h, w = frames[0].shape[:2]
        h, w = h - h % 16, w - w % 16
        K = meta["intrinsics"]
        intr = np.array([K["fx"], K["fy"], K["cx"], K["cy"]], dtype=np.float32)
        slam = DPVO(cfg, str(weights), ht=h, wd=w, viz=False)
        t0 = time.time()
        with torch.no_grad():
            for t, f in enumerate(frames):
                im = torch.from_numpy(np.ascontiguousarray(f[:h, :w])).permute(2, 0, 1).cuda()
                slam(t, im, torch.from_numpy(intr).cuda())
        wall = time.time() - t0
        poses, _tstamps = slam.terminate()
        poses = np.asarray(poses, dtype=np.float64)
        n = min(len(poses), len(truth["t"]))
        c2w = np.tile(np.eye(4), (n, 1, 1))
        from scipy.spatial.transform import Rotation
        c2w[:, :3, :3] = Rotation.from_quat(
            poses[:n, 3:7] / np.linalg.norm(poses[:n, 3:7], axis=1, keepdims=True)).as_matrix()
        c2w[:, :3, 3] = poses[:n, :3]
        q_est = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
        sc = score_clip(c2w[:, :3, 3], truth["pos"][:n], dt=1.0 / float(meta["fps"]),
                        est_quat=q_est, true_quat=truth["quat"][:n], window_s=0.5)
        rec.update({"wall_s": wall, "n_poses": int(len(poses)), "score": sc,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9})
        out = Path(C_DATA) / "poses" / "dpvo" / f"{seq}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, c2w=c2w)
        c_vol.commit()
    except torch.cuda.OutOfMemoryError:
        rec["error"] = "cuda_oom"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-1200:]
    print(json.dumps(rec, indent=1)[:1800], flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "") -> None:
    import json
    for s in [x for x in seqs.split(",") if x]:
        rec = json.loads(run_uzh_dpvo.remote(s))
        sc = rec.get("score") or {}
        print(f"== dpvo/{s}: err={rec.get('error')} dir={sc.get('accel_dir_err_deg_median')} "
              f"ratio={sc.get('accel_mag_ratio')} cov={sc.get('coverage_fraction')} "
              f"ate_sim3={sc.get('ate_sim3_m')} rot={sc.get('rel_rot_err_deg_median')}")
