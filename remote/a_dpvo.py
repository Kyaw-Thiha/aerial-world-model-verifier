"""DPVO on TII flights, on Modal: build, smoke test, and pose extraction.

DPVO must be compiled (CUDA extensions for correlation, bundle adjustment and
lietorch), so the image is built once from a CUDA *devel* base and cached by
Modal. Weights go to the Volume on first use, never into the image.

Input is the virtual pinhole view from ``src/perception/undistort.py`` (no
cv2.undistort path, so the released fisheye intrinsics are handled by our own
code and recorded), fed to DPVO as uint8 BGR cropped to a multiple of 16.

    pixi run modal run remote/a_dpvo.py --flight flight-01a-ellipse --stride 4
    pixi run modal run remote/a_dpvo.py --all --stride 4
"""
from __future__ import annotations

import modal

app = modal.App("awmv-a-dpvo")

volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)
VOLUME_MOUNT = "/vol"
DPVO_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
DPVO_ROOT = "/root/DPVO"
EIGEN_ZIP = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
MODELS_ZIP = ("https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/"
              "models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t")
FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "unzip", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.3.1", "torchvision==0.18.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torch-scatter", index_url="https://data.pyg.org/whl/torch-2.3.1+cu121.html")
    .pip_install("numpy==1.26.4", "pandas", "pyyaml", "opencv-python-headless", "einops", "pypose", "kornia",
                 "plyfile", "yacs", "evo", "tqdm", "numba", "matplotlib", "scipy")
    .run_commands(
        f"git clone https://github.com/princeton-vl/DPVO {DPVO_ROOT}",
        f"cd {DPVO_ROOT} && git checkout {DPVO_COMMIT}",
        f"cd {DPVO_ROOT} && wget -q {EIGEN_ZIP} -O eigen.zip && unzip -q eigen.zip -d thirdparty",
        f"cd {DPVO_ROOT} && pip install --no-build-isolation .",
    )
    .add_local_python_source("src")
)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=1800, cpu=2)
def fetch_weights() -> dict:
    """Download DPVO weights once into the Volume (dpvo.pth, ~100 MB)."""
    import subprocess
    from pathlib import Path

    dst = Path(VOLUME_MOUNT) / "dpvo_models"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "dpvo.pth"
    if not target.exists():
        subprocess.run(["wget", "-q", MODELS_ZIP, "-O", f"{dst}/models.zip"], check=True)
        subprocess.run(["unzip", "-o", "-q", f"{dst}/models.zip", "-d", str(dst)], check=True)
        found = list(dst.rglob("dpvo.pth"))
        if not found:
            raise FileNotFoundError(f"dpvo.pth not in {[p.name for p in dst.rglob('*')][:20]}")
        if found[0] != target:
            found[0].rename(target)
    volume.commit()
    return {"path": str(target), "bytes": target.stat().st_size}


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=3 * 3600)
def run(flight: str, stride: int = 4, fov: float = 100.0, max_frames: int = 0,
        run_tag: str = "") -> dict:
    import json
    import sys
    import time
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, DPVO_ROOT)
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    from src.datasets.tii_video import flight_parts, load_calibration
    from src.perception import undistort as U

    control, track = flight_parts(flight)
    cal = load_calibration(Path(VOLUME_MOUNT) / "tii_video" / "calibration", control, track)
    vc = U.virtual_camera(cal["K"], cal["dist"], (640, 480), fov_deg=fov)
    assert torch.cuda.is_available(), "no CUDA in this container"
    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(f"{DPVO_ROOT}/config/default.yaml")
    weights = Path(VOLUME_MOUNT) / "dpvo_models" / "dpvo.pth"
    tag = run_tag or f"{Path(flight).name}_stride{stride}_fov{int(fov)}"
    out_dir = Path(VOLUME_MOUNT) / "dpvo" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    zp = Path(VOLUME_MOUNT) / "tii_video" / control / f"camera_{flight}.zip"
    with zipfile.ZipFile(zp) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))[::stride]
        if max_frames:
            names = names[:max_frames]
        frames, ts = [], []
        for n in names:
            img = cv2.imdecode(np.frombuffer(z.read(n), dtype=np.uint8), cv2.IMREAD_COLOR)
            frames.append(U.undistort(img, vc))
            ts.append(int(Path(n).stem.split("_")[1]))
    h, w = frames[0].shape[:2]
    h, w = h - h % 16, w - w % 16
    K = vc.K.copy()
    K[0, 2] = K[0, 2] * w / vc.out_size[0]
    K[1, 2] = K[1, 2] * h / vc.out_size[1]
    K[0, 0] *= w / vc.out_size[0]
    K[1, 1] *= h / vc.out_size[1]
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)

    slam = DPVO(cfg, str(weights), ht=h, wd=w, viz=False)
    t0 = time.time()
    losses = []
    for t, f in enumerate(frames):
        im = torch.from_numpy(np.ascontiguousarray(f[:h, :w])).permute(2, 0, 1).cuda()
        slam(t, im, torch.from_numpy(intr).cuda())
        if hasattr(slam, "pg") and hasattr(slam.pg, "n"):
            losses.append(int(slam.pg.n))
    wall = time.time() - t0
    poses, tstamps = slam.terminate()
    poses = np.asarray(poses, dtype=np.float64)
    peak = torch.cuda.max_memory_allocated() / 1e9
    rec = {"flight": flight, "stride": stride, "fov_deg": fov, "n_frames_fed": len(frames),
           "n_poses": int(len(poses)), "seconds": wall,
           "fps": len(frames) / wall if wall else 0.0, "peak_gpu_gb": peak,
           "virtual_valid_frac": vc.valid_fraction, "image_hw": [h, w],
           "intrinsics_fx_fy_cx_cy": intr.tolist(), "patch_counts": losses[::50],
           "dpvo_commit": DPVO_COMMIT,
           "pose_layout": "tx ty tz qx qy qz qw (DPVO convention, up to scale)"}
    np.savez_compressed(out_dir / "poses.npz", poses=poses,
                        tstamps=np.asarray(tstamps, dtype=np.float64),
                        frame_ts_us=np.asarray(ts[:len(poses)], dtype=np.int64),
                        intrinsics=intr)
    (out_dir / "summary.json").write_text(json.dumps(rec, indent=2))
    volume.commit()
    print(json.dumps(rec, indent=2)[:1500], flush=True)
    return rec


@app.local_entrypoint()
def main(flight: str = "flight-01a-ellipse", stride: int = 4, fov: float = 100.0,
         max_frames: int = 0, all: bool = False, fetch: bool = True):
    import json

    if fetch:
        print("weights:", json.dumps(fetch_weights.remote()))
    flights = [f for v in FLIGHTS.values() for f in v] if all else [flight]
    for f in flights:
        print(json.dumps(run.remote(f, stride, fov, max_frames), indent=2)[:800])
