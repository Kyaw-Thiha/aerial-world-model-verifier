"""Track F, stage 1: MASt3R-SLAM on the ten UZH-FPV sequences, Kalibr K supplied.

MASt3R-SLAM (Murai et al.) is the SLAM system for stage 1. It is given the published Kalibr
pinhole K on the command line, so — unlike ViPE — it runs on the same calibration footing as
DPVO in Track C's bake-off.

    pixi run modal run remote/f_mast3r.py --seqs indoor_forward_5
    pixi run modal run remote/f_mast3r.py

Poses land on ``awmv-f-data`` at ``poses/mast3r/<seq>.npz`` as ``c2w`` (N, 4, 4), interpolated
onto the 150-frame grid when the tracker drops frames.
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-f-mast3r")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
f_vol = modal.Volume.from_name("awmv-f-data", create_if_missing=True)
C_DATA = "/c-data"
F_DATA = "/f-data"

SEQS = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9"]

mast3r_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "wget")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "wheel", "setuptools", "ninja", "packaging", "cython")
    # no GPU is visible during a Modal image build, so torch cannot infer the arch list
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM /opt/mast3r-slam",
        "cd /opt/mast3r-slam && pip install -e thirdparty/mast3r",
        "cd /opt/mast3r-slam && pip install -e thirdparty/in3d",
        "cd /opt/mast3r-slam && pip install --no-build-isolation -e .",
        "mkdir -p /opt/mast3r-slam/checkpoints && cd /opt/mast3r-slam/checkpoints && "
        "wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && "
        "wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth && "
        "wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
    )
)


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


def _truth_meta(seq: str) -> dict:
    from pathlib import Path

    meta = json.loads((Path(C_DATA) / "uzh" / seq / "meta.json").read_text())
    return {"fps": float(meta["fps_measured"]), "intrinsics": meta["intrinsics_pinhole"]}


def _write_calib(path: str, K: dict) -> str:
    from pathlib import Path

    Path(path).write_text(
        f"width: {int(K['w'])}\nheight: {int(K['h'])}\n"
        f"calibration: [{K['fx']}, {K['fy']}, {K['cx']}, {K['cy']}]\n")
    return path


def _save(poses, seq: str, extra: dict) -> dict:
    import numpy as np
    from pathlib import Path

    out = Path(F_DATA) / "poses" / "mast3r"
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=np.float64)
    np.savez_compressed(out / f"{seq}.npz", c2w=poses)
    f_vol.commit()
    return {"seq": seq, "method": "mast3r", "n_poses": int(len(poses)),
            "finite": bool(np.isfinite(poses).all()),
            "path_len_units": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                   axis=1).sum()) if len(poses) > 1 else 0.0,
            **extra}


@app.function(image=mast3r_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def mast3r_run(seqs: list[str]) -> str:
    import subprocess
    import time
    from pathlib import Path

    import numpy as np
    from scipy.spatial.transform import Rotation

    recs = []
    for seq in seqs:
        frames = _stage(seq)
        meta = _truth_meta(seq)
        calib = _write_calib("/tmp/intrinsics.yaml", meta["intrinsics"])
        t0 = time.time()
        cmd = ["python", "main.py", "--dataset", frames, "--config", "config/base.yaml",
               "--calib", calib, "--save-as", f"f01_{seq}"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/mast3r-slam",
                           env={"PYTHONPATH": "/opt/mast3r-slam", "PATH": "/usr/bin:/bin",
                                "HOME": "/root", "MPLBACKEND": "Agg",
                                "TORCH_CUDA_ARCH_LIST": "8.0"})
        rec = {"seq": seq, "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
               "stderr_tail": r.stderr[-2000:], "stdout_tail": r.stdout[-1200:]}
        if r.returncode != 0:
            recs.append(rec)
            continue
        trajs = sorted(Path("/opt/mast3r-slam/logs").rglob("*.txt"))
        if not trajs:
            rec["error"] = "no trajectory txt"
            recs.append(rec)
            continue
        traj = trajs[-1]
        rows = [ln.split() for ln in traj.read_text().splitlines() if ln.strip()]
        rows = [x for x in rows if len(x) >= 8]
        if not rows:
            rec["error"] = "empty trajectory"
            recs.append(rec)
            continue
        data = np.asarray(rows, dtype=np.float64)
        ts, pos, quat = data[:, 0], data[:, 1:4], data[:, 4:8]
        R = Rotation.from_quat(quat).as_matrix()
        order = np.argsort(ts)
        c2w = np.tile(np.eye(4), (len(order), 1, 1))
        c2w[:, :3, :3] = R[order]
        c2w[:, :3, 3] = pos[order]
        t_norm = ts[order] - ts[order][0]
        t_frames = np.arange(len(order)) / meta["fps"]
        tgt = np.clip(t_frames, t_norm[0], t_norm[-1])
        flat = np.column_stack([np.interp(tgt, t_norm, c2w[:, i, j])
                                for i in range(3) for j in range(4)])
        c2w_out = np.tile(np.eye(4), (len(t_frames), 1, 1))
        c2w_out[:, :3, :4] = flat.reshape(-1, 3, 4)
        rec.update(_save(c2w_out, seq, {"traj_file": str(traj), "n_raw": int(len(order)),
                                        "raw_dt_s": float(np.median(np.diff(t_norm)))
                                        if len(t_norm) > 1 else None}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(seqs: str = ""):
    chosen = [s for s in seqs.split(",") if s] or SEQS
    print(mast3r_run.remote(chosen))
