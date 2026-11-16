"""Track F, stage 1: other path extractors on the ten UZH-FPV sequences.

Two systems the bake-off never ran:

  vipe     ViPE (nv-tlabs), pip package ``nvidia-vipe`` 1.2.0, default pinhole pipeline with
           its own SharedK intrinsics. It is run *without* the Kalibr intrinsics on purpose:
           the default pipeline estimates them, and that is the axis C07 isolates.
  mast3r   MASt3R-SLAM with the published Kalibr pinhole K supplied on the command line.

MegaSaM is not in this app. Its camera-tracking weights (RAFT ``raft-things.pth``) are
distributed only through a Google Drive folder, so a container cannot fetch them
non-interactively; that is recorded as a blocker rather than papered over. DROID-SLAM, which
MegaSaM builds on, has the same distribution problem for ``droid.pth``.

Everything reads the prepared UZH clips already on the ``awmv-c-data`` volume
(``uzh/<seq>/frames/*.png`` 150 frames at 15.07 Hz, plus ``truth.npz``) and writes
camera-to-world poses to ``awmv-f-data`` under ``poses/<method>/<seq>.npz``.

    pixi run modal run remote/f_extractors.py --probe ost           # output-format probe
    pixi run modal run remote/f_extractors.py --method vipe        # ten sequences
    pixi run modal run remote/f_extractors.py --method mast3r --seqs indoor_forward_5
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-f-extractors")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
f_vol = modal.Volume.from_name("awmv-f-data", create_if_missing=True)
C_DATA = "/c-data"
F_DATA = "/f-data"

SEQS = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9"]

# --------------------------------------------------------------------------- images
vipe_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg", "libopenexr-dev", "build-essential")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy<2.3", "wheel", "setuptools", "ninja", "packaging")
    .run_commands(
        "CUDA_HOME=/usr/local/cuda MAX_JOBS=8 pip install --no-build-isolation nvidia-vipe==1.2.0",
    )
    .env({"HF_HOME": f"{F_DATA}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "CUDA_HOME": "/usr/local/cuda"})
)

mast3r_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "wget")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "wheel", "setuptools", "ninja", "packaging")
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
    .env({"HF_HOME": f"{F_DATA}/hf", "PYTHONUNBUFFERED": "1"})
)


# --------------------------------------------------------------------------- helpers
def _stage(seq: str, limit: int = 150) -> str:
    """Copy the prepared frames to a local directory (volume reads are slow in a loop)."""
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
    import json as _json
    from pathlib import Path

    meta = _json.loads((Path(C_DATA) / "uzh" / seq / "meta.json").read_text())
    truth = __import__("numpy").load(Path(C_DATA) / "uzh" / seq / "truth.npz")
    return {"fps": float(meta["fps_measured"]), "n_truth": int(len(truth["t"])),
            "intrinsics": meta["intrinsics_pinhole"]}


def _save(poses, seq: str, method: str, extra: dict) -> dict:
    """Write a c2w array (N, 4, 4) to the F volume; return a small record."""
    import numpy as np

    from pathlib import Path

    out = Path(F_DATA) / "poses" / method
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=np.float64)
    np.savez_compressed(out / f"{seq}.npz", c2w=poses)
    f_vol.commit()
    return {"seq": seq, "method": method, "n_poses": int(len(poses)),
            "finite": bool(np.isfinite(poses).all()),
            "path_len_units": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                   axis=1).sum()) if len(poses) > 1 else 0.0,
            "path": str(out / f"{seq}.npz"), **extra}


# --------------------------------------------------------------------------- ViPE
@app.function(image=vipe_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def vipe_probe(seq: str) -> str:
    """Run ViPE on one sequence and print where it put the poses and in what shape."""
    import subprocess
    import time
    from pathlib import Path

    frames = _stage(seq)
    out = Path("/tmp/vipe_out")
    t0 = time.time()
    cmd = ["vipe", "infer", "--image-dir", frames, "-o", str(out), "-p", "default"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"seq": seq, "cmd": " ".join(cmd), "returncode": r.returncode,
           "wall_s": round(time.time() - t0, 1), "stdout_tail": r.stdout[-3000:],
           "stderr_tail": r.stderr[-3000:]}
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()][:80] if out.exists() else []
    rec["files"] = files
    if out.exists():
        import numpy as np
        for name in ("pose",):
            d = out / name
            if d.exists():
                for p in sorted(d.glob("*"))[:3]:
                    try:
                        z = np.load(p)
                        rec.setdefault("arrays", {})[str(p)] = {
                            k: list(np.asarray(z[k]).shape) for k in z.files}
                    except Exception as exc:  # noqa: BLE001
                        rec.setdefault("arrays", {})[str(p)] = f"unreadable: {exc}"
    return json.dumps(rec, indent=1)


@app.function(image=vipe_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def vipe_run(seqs: list[str]) -> str:
    """Run ViPE's default pinhole pipeline on prepared UZH sequences; save c2w poses."""
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    recs = []
    for seq in seqs:
        frames = _stage(seq)
        meta = _truth_meta(seq)
        out = Path("/tmp/vipe_out") / seq
        t0 = time.time()
        cmd = ["vipe", "infer", "--image-dir", frames, "-o", str(out), "-p", "default"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rec = {"seq": seq, "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
               "fps": meta["fps"], "n_frames": 150,
               "stderr_tail": r.stderr[-1500:]}
        if r.returncode != 0:
            recs.append(rec)
            continue
        npzs = sorted((out / "pose").glob("*.npz"))
        if not npzs:
            rec["error"] = "no pose npz"
            recs.append(rec)
            continue
        z = np.load(npzs[0])
        key = "pose" if "pose" in z.files else ("c2w" if "c2w" in z.files else z.files[0])
        arr = np.asarray(z[key], dtype=np.float64)
        if arr.shape[-2:] == (3, 4):
            c2w = np.tile(np.eye(4), (arr.shape[0], 1, 1))
            c2w[:, :3, :4] = arr
        elif arr.shape[-2:] == (4, 4):
            c2w = arr
        else:
            rec["error"] = f"unexpected pose shape {arr.shape}"
            recs.append(rec)
            continue
        rec.update(_save(c2w, seq, "vipe", {"pose_key": key, "raw_shape": list(arr.shape)}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


# --------------------------------------------------------------------------- MASt3R-SLAM
def _write_calib(path: str, K: dict) -> str:
    text = (f"width: {int(K['w'])}\nheight: {int(K['h'])}\n"
            f"calibration: [{K['fx']}, {K['fy']}, {K['cx']}, {K['cy']}]\n")
    from pathlib import Path
    Path(path).write_text(text)
    return path


@app.function(image=mast3r_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def mast3r_run(seqs: list[str]) -> str:
    """MASt3R-SLAM with the published Kalibr pinhole K; save c2w poses in frame order."""
    import re
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

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
                                "HOME": "/root", "MPLBACKEND": "Agg"})
        rec = {"seq": seq, "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
               "fps": meta["fps"], "stderr_tail": r.stderr[-1500:]}
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
        # TUM: timestamp tx ty tz qx qy qz qw (camera-to-world)
        from scipy.spatial.transform import Rotation
        ts, pos, quat = data[:, 0], data[:, 1:4], data[:, 4:8]
        R = Rotation.from_quat(quat).as_matrix()
        order = np.argsort(ts)
        c2w = np.tile(np.eye(4), (len(order), 1, 1))
        c2w[:, :3, :3] = R[order]
        c2w[:, :3, 3] = pos[order]
        # interpolate onto the frame grid if the tracker dropped frames
        t_frames = np.arange(len(order)) / meta["fps"]
        t_norm = (ts[order] - ts[order][0])
        tgt = np.clip(t_frames, t_norm[0], t_norm[-1])
        interp = np.column_stack([np.interp(tgt, t_norm, c2w[:, i, j])
                                  for i in range(3) for j in range(4)])
        c2w_out = np.tile(np.eye(4), (len(t_frames), 1, 1))
        c2w_out[:, :3, :4] = interp.reshape(-1, 3, 4)
        rec.update(_save(c2w_out, seq, "mast3r",
                         {"traj_file": str(traj), "n_raw": int(len(order)),
                          "raw_dt_s": float(np.median(np.diff(t_norm))) if len(t_norm) > 1 else None}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(method: str = "", seqs: str = "", probe: str = ""):
    chosen = [s for s in seqs.split(",") if s] or SEQS
    if probe:
        print(vipe_probe.remote(probe))
        return
    if method == "vipe":
        print(vipe_run.remote(chosen))
    elif method == "mast3r":
        print(mast3r_run.remote(chosen))
    else:
        raise SystemExit("pass --method vipe|mast3r, or --probe <sequence>")
