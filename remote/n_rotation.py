"""Track N, Part 0: DPVO camera rotations for the rotation-propagation pre-test.

Runs the same DPVO build and settings as Track A/Track C (commit ``0ac95b6``, 48 patches,
removal 12, optimisation 8, lifetime 5) on two populations:

* ``uzh`` -- the 14 prepared UZH-FPV pinhole sequences on the ``awmv-e-data`` volume, all 450
  frames each (30 s at 15.07 Hz). Track C's stored poses cover only 150 frames, which cannot
  reach a 10 s horizon; this re-run extends them. Read-only mount; nothing is written there.
* ``sim`` -- the 80 feasible development clips of Track I's frozen simulated split, videos
  uploaded byte-identically to ``awmv-n-data/sim/videos`` from Track L's verified copy.

Outputs (on ``awmv-n-data``, pulled locally for scoring):

    poses/uzh/<sequence>.npz   c2w (n,4,4), t, n_input, n_pose, wall_s, fps, intr, ...
    poses/sim/<clip_id>.npz    same
    runs/<kind>__<unit>.json   the same summary without the arrays

    modal run remote/n_rotation.py --which uzh
    modal run remote/n_rotation.py --which sim
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-n-rotation")

n_vol = modal.Volume.from_name("awmv-n-data", create_if_missing=True)
e_vol = modal.Volume.from_name("awmv-e-data")
a_vol = modal.Volume.from_name("awmv-a-data")

N_DATA = "/n-data"
E_DATA = "/e-data"
A_DATA = "/a-data"
DPVO_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
DPVO_ROOT = "/root/DPVO"
EIGEN_ZIP = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
MODELS_ZIP = ("https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/"
              "models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t")

# The 14 prepared sequences on awmv-e-data/uzh. Gate stratum: the three outdoor_forward rigs;
# the rest are reported descriptively (pre-registration configs/n01_pretest_prereg.yaml).
UZH = [
    "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5",
    "indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
    "indoor_forward_9", "indoor_forward_10",
    "indoor_45_2", "indoor_45_4", "indoor_45_9", "indoor_45_13", "outdoor_45_1",
]

# Render intrinsics for the simulated clips: 640x360, 90 deg horizontal field (L01).
SIM_INTR = (320.0, 320.0, 320.0, 180.0)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "unzip", "build-essential", "clang", "libgl1", "libglib2.0-0")
    .pip_install("wheel", "setuptools")
    .pip_install("torch==2.3.0", "torchvision==0.18.0",
                 index_url="https://download.pytorch.org/whl/cu121")
    .run_commands("pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html")
    .pip_install("numpy==1.26.4", "pandas", "pyyaml", "einops",
                 "pypose", "kornia", "plyfile", "yacs", "evo", "tqdm", "numba", "matplotlib",
                 "scipy")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6"})
    .run_commands(
        f"git clone https://github.com/princeton-vl/DPVO {DPVO_ROOT}",
        f"cd {DPVO_ROOT} && git checkout {DPVO_COMMIT}",
        f"cd {DPVO_ROOT} && wget -q {EIGEN_ZIP} -O eigen.zip && unzip -q eigen.zip -d thirdparty",
        f"cd {DPVO_ROOT} && pip install --no-build-isolation .",
    )
    # installed after DPVO's own setup so its resolver cannot drop it
    .pip_install("opencv-python-headless==4.10.0.84")
)


def _run_dpvo(frames, intr: tuple[float, float, float, float], fps: float, unit: str,
              kind: str) -> dict:
    """Run DPVO over a list of BGR/grey numpy frames; return camera-to-world poses."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, DPVO_ROOT)
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    h, w = frames[0].shape[:2]
    h, w = h - h % 16, w - w % 16
    fx, fy, cx, cy = [float(v) for v in intr]
    K = np.array([fx, fy, cx, cy], dtype=np.float32)
    weights = Path(A_DATA) / "dpvo_models" / "dpvo.pth"
    cfg.merge_from_file(f"{DPVO_ROOT}/config/default.yaml")
    cfg.PATCHES_PER_FRAME = 48
    cfg.REMOVAL_WINDOW = 12
    cfg.OPTIMIZATION_WINDOW = 8
    cfg.PATCH_LIFETIME = 5
    slam = DPVO(cfg, str(weights), ht=h, wd=w, viz=False)
    import time
    t0 = time.time()
    with torch.no_grad():
        for t, f in enumerate(frames):
            arr = np.ascontiguousarray(f[:h, :w])
            if arr.ndim == 2:
                im = torch.from_numpy(arr)[None].repeat(3, 1, 1)
            else:
                im = torch.from_numpy(arr).permute(2, 0, 1)
                if im.shape[0] == 1:
                    im = im.repeat(3, 1, 1)
            slam(t, im.cuda(), torch.from_numpy(K).cuda())
    wall = time.time() - t0
    poses, _ = slam.terminate()
    poses = np.asarray(poses, dtype=np.float64)
    from scipy.spatial.transform import Rotation
    c2w = np.tile(np.eye(4), (len(poses), 1, 1))
    if len(poses):
        q = poses[:, 3:7]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        c2w[:, :3, :3] = Rotation.from_quat(q).as_matrix()
        c2w[:, :3, 3] = poses[:, :3]
    return {"c2w": c2w, "t": np.arange(len(poses), dtype=np.float64) / fps,
            "n_input": int(len(frames)), "n_pose": int(len(poses)), "wall_s": float(wall),
            "fps": float(fps), "intr": [fx, fy, cx, cy], "kind": kind, "unit": unit,
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9)}


@app.function(image=image, volumes={E_DATA: e_vol, N_DATA: n_vol, A_DATA: a_vol},
              gpu="A10", timeout=3600, max_containers=4)
def uzh(seq: str, max_frames: int = 450, force: bool = False) -> dict:
    """DPVO on one prepared UZH pinhole sequence (read-only from awmv-e-data)."""
    import cv2
    import numpy as np

    out_npz = Path(N_DATA) / "poses" / "uzh" / f"{seq}.npz"
    out_json = Path(N_DATA) / "runs" / f"uzh__{seq}.json"
    if out_npz.exists() and not force:
        return json.loads(out_json.read_text())
    meta = json.loads((Path(E_DATA) / "uzh" / seq / "meta.json").read_text())
    frame_dir = Path(E_DATA) / "uzh" / seq / "frames_pinhole"
    paths = sorted(frame_dir.glob("*.png"))[:max_frames]
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    if not frames or frames[0] is None:
        raise RuntimeError(f"{seq}: no frames under {frame_dir}")
    ip = meta["intrinsics_pinhole"]
    rec = _run_dpvo(frames, (ip["fx"], ip["fy"], ip["cx"], ip["cy"]),
                    float(meta["fps_measured"]), seq, "uzh")
    rec.update({"sequence": seq, "rig": meta.get("rig"), "environment": meta.get("environment"),
                "dpvo_commit": DPVO_COMMIT})
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, c2w=rec.pop("c2w"), t=rec.pop("t"),
                        n_input=rec["n_input"], n_pose=rec["n_pose"], fps=rec["fps"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=1))
    n_vol.commit()
    return rec


@app.function(image=image, volumes={N_DATA: n_vol, A_DATA: a_vol},
              gpu="A10", timeout=3600, max_containers=4)
def sim(clip_id: str, force: bool = False) -> dict:
    """DPVO on one simulated clip's mp4 (uploaded to awmv-n-data/sim/videos)."""
    import cv2
    import numpy as np

    out_npz = Path(N_DATA) / "poses" / "sim" / f"{clip_id}.npz"
    out_json = Path(N_DATA) / "runs" / f"sim__{clip_id}.json"
    if out_npz.exists() and not force:
        return json.loads(out_json.read_text())
    video = Path(N_DATA) / "sim" / "videos" / f"{clip_id}.mp4"
    cap = cv2.VideoCapture(str(video))
    frames, fps = [], float(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"{clip_id}: no frames decoded from {video}")
    if not fps or fps < 1:
        raise RuntimeError(f"{clip_id}: unreadable fps")
    rec = _run_dpvo(frames, SIM_INTR, fps, clip_id, "sim")
    rec.update({"clip_id": clip_id, "video": str(video), "dpvo_commit": DPVO_COMMIT})
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, c2w=rec.pop("c2w"), t=rec.pop("t"),
                        n_input=rec["n_input"], n_pose=rec["n_pose"], fps=rec["fps"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=1))
    n_vol.commit()
    return rec


@app.function(image=image, volumes={N_DATA: n_vol, A_DATA: a_vol},
              gpu="A10", timeout=3600, max_containers=6)
def gen(clip: str, fov_deg: float = 90.0, force: bool = False) -> dict:
    """DPVO on one generated clip (H04 SolarWM), native 864x480, assumed focal."""
    import cv2
    import numpy as np

    tag = f"fov{int(round(fov_deg))}"
    out_npz = Path(N_DATA) / "poses" / "gen" / tag / f"{clip}.npz"
    out_json = Path(N_DATA) / "runs" / f"gen__{tag}__{clip}.json"
    if out_npz.exists() and not force:
        return json.loads(out_json.read_text())
    video = Path(N_DATA) / "gen" / "videos" / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(video))
    frames, fps = [], float(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"{clip}: no frames decoded")
    h, w = frames[0].shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    rec = _run_dpvo(frames, (f, f, w / 2.0, h / 2.0), fps, clip, "gen")
    rec.update({"clip_id": clip, "fov_deg": fov_deg, "video": str(video),
                "dpvo_commit": DPVO_COMMIT})
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, c2w=rec.pop("c2w"), t=rec.pop("t"),
                        n_input=rec["n_input"], n_pose=rec["n_pose"], fps=rec["fps"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=1))
    n_vol.commit()
    return rec


@app.function(image=image, volumes={E_DATA: e_vol, N_DATA: n_vol, A_DATA: a_vol},
              gpu="A10", timeout=3600, max_containers=6)
def uzh_edit(clip_id: str, seq: str, force: bool = False) -> dict:
    """DPVO on one Track-E v3 edited clip (320x240 raw crop -> undistorted pinhole)."""
    import cv2
    import numpy as np

    out_npz = Path(N_DATA) / "poses" / "uzh_edits" / f"{clip_id}.npz"
    out_json = Path(N_DATA) / "runs" / f"uzh_edit__{clip_id}.json"
    if out_npz.exists() and not force:
        return json.loads(out_json.read_text())
    meta = json.loads((Path(E_DATA) / "uzh" / seq / "meta.json").read_text())
    ip = meta["intrinsics_kb4"]
    fx, fy = float(ip["fx"]), float(ip["fy"])
    # Track E's v3 clips are a fixed centred 0.5 crop of the 640x480 raw frame
    cx, cy = float(ip["cx"]) - 160.0, float(ip["cy"]) - 120.0
    K_in = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array(ip["k"], dtype=np.float64)
    K_out = np.array([[fx, 0, 160.0], [0, fy, 120.0], [0, 0, 1.0]])
    frame_dir = Path(N_DATA) / "uzh_clips" / seq / clip_id / "frames"
    paths = sorted(frame_dir.glob("*.png"))
    frames = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        und = cv2.fisheye.undistortImage(im, K_in, D, Knew=K_out, new_size=(320, 240))
        frames.append(und)
    if not frames:
        raise RuntimeError(f"{clip_id}: no frames under {frame_dir}")
    fps = 15.07
    rec = _run_dpvo(frames, (fx, fy, 160.0, 120.0), fps, clip_id, "uzh_edit")
    rec.update({"clip_id": clip_id, "sequence": seq, "dpvo_commit": DPVO_COMMIT,
                "undistort": "KB4 crop 0.5 -> pinhole 320x240, same focal"})
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, c2w=rec.pop("c2w"), t=rec.pop("t"),
                        n_input=rec["n_input"], n_pose=rec["n_pose"], fps=rec["fps"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=1))
    n_vol.commit()
    return rec


@app.local_entrypoint()
def main(which: str = "all", limit: int = 0, ids: str = "", seq: str = "", fov: float = 90.0):
    import csv

    if which in ("all", "uzh"):
        seqs = [seq] if seq else UZH
        for r in uzh.map(seqs):
            print("UZH", r)
    if which == "gen":
        import csv as _csv
        with open(Path("results/n04/clips.csv")) as fh:
            vids = [r["clip"] for r in _csv.DictReader(fh)]
        if ids:
            vids = [x for x in ids.split(",") if x]
        print(f"gen clips: {len(vids)} fov {fov}")
        for r in gen.map(vids, kwargs={"fov_deg": fov}):
            print("GEN", r)
    if which == "uzh_edit":
        import csv as _csv
        with open(Path("results/n02b/clips.csv")) as fh:
            rows = [(r["clip_id"], r["seq"]) for r in _csv.DictReader(fh)]
        print(f"uzh edit clips: {len(rows)}")
        for r in uzh_edit.starmap(rows):
            print("EDIT", r)
    if which in ("all", "sim"):
        if ids:
            clip_ids = [x for x in ids.split(",") if x]
        else:
            path = Path("results/n01_pretest/sim_population.csv")
            with open(path) as fh:
                clip_ids = [r["clip_id"] for r in csv.DictReader(fh)]
        clip_ids = sorted(clip_ids)
        if limit:
            clip_ids = clip_ids[:limit]
        print(f"sim clips: {len(clip_ids)}")
        for r in sim.map(clip_ids):
            print("SIM", r)
