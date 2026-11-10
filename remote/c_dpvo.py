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


def _dpvo_score(seq: str, max_frames: int = 150, patches: int = 48,
                fx_scale: float = 1.0, intr_override: list | None = None,
                commit: bool = False) -> dict:
    import json
    import sys
    import time
    import traceback
    from pathlib import Path  # noqa: F401

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
        if intr_override is not None:
            fx, fy, cx, cy = [float(v) for v in intr_override]
        else:
            fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        intr = np.array([fx * fx_scale, fy * fx_scale, cx, cy], dtype=np.float32)
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
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "fx_scale": fx_scale, "intr_used": intr.tolist(),
                    "intr_source": "override" if intr_override is not None else "calib"})
        if commit:
            out = Path(C_DATA) / "poses" / "dpvo" / f"{seq}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, c2w=c2w)
            c_vol.commit()
    except torch.cuda.OutOfMemoryError:
        rec["error"] = "cuda_oom"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-1200:]
    return rec


@app.function(image=image, gpu="A10G", volumes={C_DATA: c_vol, "/a-data": a_vol},
              timeout=3600, cpu=2)
def run_uzh_dpvo(seq: str, max_frames: int = 150, patches: int = 48) -> str:
    rec = _dpvo_score(seq, max_frames, patches, commit=True)
    print(json.dumps(rec, indent=1)[:1500], flush=True)
    return json.dumps(rec)


@app.function(image=image, gpu="A10G", volumes={C_DATA: c_vol, "/a-data": a_vol},
              timeout=3600, cpu=2)
def calib_matrix(seqs: list[str], scales: list[float], max_frames: int = 150,
                 intr_table: str = "") -> str:
    """DPVO with perturbed focal length and/or estimated intrinsics, per clip."""
    import json
    table = json.loads(intr_table) if intr_table else {}
    out = []
    for seq in seqs:
        overrides = [(f"fx*{s:g}", s, None) for s in scales]
        if seq in table:
            for src, K in table[seq].items():
                if K:
                    overrides.append((src, 1.0, K))
        for name, s, ov in overrides:
            rec = _dpvo_score(seq, max_frames, fx_scale=s, intr_override=ov)
            rec["condition"] = name
            out.append(rec)
            sc = rec.get("score") or {}
            print(f"[{seq} {name}] err={rec.get('error')} "
                  f"dir={sc.get('accel_dir_err_deg_median')} "
                  f"ratio={sc.get('accel_mag_ratio')} cov={sc.get('coverage_fraction')}",
                  flush=True)
    return json.dumps(out)


@app.function(image=image, gpu="A10G", volumes={C_DATA: c_vol, "/a-data": a_vol},
              timeout=3600, cpu=2)
def anwm_attribution(regimes: str = "A", max_clips: int = 0) -> str:
    """DPVO on ANWM clips: generated frames vs same-start ground-truth frames.

    Intrinsics are the manifest's f = 112 px at 224x224; one frame per commanded step at the
    documented 1 s/step, so a 3-5 m baseline sits between consecutive frames.
    """
    import json
    import sys
    import time
    import traceback
    from pathlib import Path as P

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, DPVO_ROOT)
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    from src.pathrec.evaluate import score_clip

    root = P("/a-data") / "anwm_clips"
    want = {x.strip() for x in regimes.split(",") if x.strip()}
    clips = []
    for man in sorted(root.rglob("manifest.json")):
        meta = json.loads(man.read_text())
        if want and meta.get("regime") not in want:
            continue
        clips.append((man.parent, meta))
    if max_clips:
        clips = clips[:max_clips]

    def track(frames):
        cfg.merge_from_file(f"{DPVO_ROOT}/config/default.yaml")
        cfg.PATCHES_PER_FRAME = 48
        cfg.REMOVAL_WINDOW = 6
        cfg.OPTIMIZATION_WINDOW = 6
        cfg.PATCH_LIFETIME = 4
        h, w = frames[0].shape[:2]
        h, w = h - h % 16, w - w % 16
        intr = np.array([112.0, 112.0, 112.0, 112.0], dtype=np.float32)
        slam = DPVO(cfg, str(P("/a-data") / "dpvo_models" / "dpvo.pth"), ht=h, wd=w, viz=False)
        with torch.no_grad():
            for t, f in enumerate(frames):
                im = torch.from_numpy(np.ascontiguousarray(f[:h, :w])).permute(2, 0, 1).cuda()
                slam(t, im, torch.from_numpy(intr).cuda())
        poses, _ = slam.terminate()
        return np.asarray(poses, dtype=np.float64)

    out = []
    for d, meta in clips:
        rec = {"model": "dpvo", "clip": str(d.relative_to(root)), "regime": meta.get("regime")}
        try:
            gen = [cv2.imread(str(p)) for p in sorted((d / "frames").glob("*.png"))]
            cmd = np.asarray(meta["commanded_path"]["positions"], dtype=np.float64)
            n = len(gen)
            truth = cmd[1:n + 1] if len(cmd) == n + 1 else cmd[:n]
            for cond, frames in (("generated", gen),
                                 ("gt", [cv2.imread(str(p)) for p in sorted((d / "gt").glob("*.png"))])):
                if not frames:
                    continue
                t0 = time.time()
                poses = track(frames)
                m = min(len(poses), len(truth))
                rec[cond] = score_clip(poses[:m, :3], truth[:m], dt=1.0, window_s=5.0)
                rec[cond]["wall_s"] = time.time() - t0
                rec[cond]["n_poses"] = int(len(poses))
        except torch.cuda.OutOfMemoryError:
            rec["error"] = "cuda_oom"
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-800:]
        out.append(rec)
        print(f"[{rec['clip']}] err={rec.get('error')} "
              f"gen_dir={(rec.get('generated') or {}).get('accel_dir_err_deg_median')} "
              f"gt_dir={(rec.get('gt') or {}).get('accel_dir_err_deg_median')}", flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(seqs: str = "", calib: bool = False, scales: str = "0.8,0.9,1.0,1.1,1.2",
         intr_json: str = "", out_json: str = "", anwm: bool = False,
         regimes: str = "A") -> None:
    import json
    import os
    seq_list = [x for x in seqs.split(",") if x]
    if anwm:
        out = json.loads(anwm_attribution.remote(regimes))
        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            open(out_json, "w").write(json.dumps(out, indent=1))
        for rec in out:
            g, q = rec.get("generated") or {}, rec.get("gt") or {}
            print(f"== dpvo/{rec['clip']}: err={rec.get('error')} "
                  f"gen dir={g.get('accel_dir_err_deg_median')} ratio={g.get('accel_mag_ratio')} "
                  f"| gt dir={q.get('accel_dir_err_deg_median')} ratio={q.get('accel_mag_ratio')}",
                  flush=True)
        return
    if calib:
        table = {}
        if intr_json and os.path.exists(intr_json):
            rows = json.loads(open(intr_json).read())
            for r in rows:
                K = r.get("K")
                # DA3/VGGT give a 3x3 in pixel units of their own raster; scale to the
                # 640x480 prepared frames by the width ratio recorded in the row
                if K:
                    fx, fy = float(K[0][0]), float(K[1][1])
                    cx, cy = float(K[0][2]), float(K[1][2])
                    w_src = r.get("proc_w")
                    if w_src:
                        f = 640.0 / float(w_src)
                        fx, fy, cx, cy = fx * f, fy * f, cx * f, cy * f
                    table.setdefault(r["seq"], {})[r["model"]] = [fx, fy, cx, cy]
        out = json.loads(calib_matrix.remote(seq_list, [float(x) for x in scales.split(",")],
                                             intr_table=json.dumps(table)))
        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            open(out_json, "w").write(json.dumps(out, indent=1))
        for rec in out:
            sc = rec.get("score") or {}
            print(f"== {rec['seq']} {rec.get('condition')}: err={rec.get('error')} "
                  f"dir={sc.get('accel_dir_err_deg_median')} ratio={sc.get('accel_mag_ratio')}",
                  flush=True)
        return
    for s in seq_list:
        rec = json.loads(run_uzh_dpvo.remote(s))
        sc = rec.get("score") or {}
        print(f"== dpvo/{s}: err={rec.get('error')} dir={sc.get('accel_dir_err_deg_median')} "
              f"ratio={sc.get('accel_mag_ratio')} cov={sc.get('coverage_fraction')} "
              f"ate_sim3={sc.get('ate_sim3_m')} rot={sc.get('rel_rot_err_deg_median')}")
