"""Track F, stage 1: ViPE on the ten UZH-FPV sequences.

ViPE (nv-tlabs, ``nvidia-vipe`` 1.2.0) is run with its **own** SharedK intrinsics — the default
pinhole pipeline estimates them — which is deliberate: Track C's C07 showed that supplying
the Kalibr calibration is what separates DPVO from the feed-forward extractors, so a ViPE run
without it measures the estimator, not the calibration.

    pixi run modal run remote/f_vipe.py --probe indoor_forward_5   # build + format probe
    pixi run modal run remote/f_vipe.py --seqs indoor_forward_5   # one sequence
    pixi run modal run remote/f_vipe.py                            # ten sequences

Poses land on the ``awmv-f-data`` volume at ``poses/vipe/<seq>.npz`` as ``c2w`` (N, 4, 4).
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-f-vipe")

c_vol = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
f_vol = modal.Volume.from_name("awmv-f-data", create_if_missing=True)
C_DATA = "/c-data"
F_DATA = "/f-data"

SEQS = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9"]

vipe_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg", "libopenexr-dev", "build-essential")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy<2.3", "wheel", "setuptools", "ninja", "packaging")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "MAX_JOBS=8 pip install --no-build-isolation nvidia-vipe==1.2.0",
    )
    .env({"HF_HOME": f"{F_DATA}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
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


def _save(poses, seq: str, extra: dict) -> dict:
    import numpy as np
    from pathlib import Path

    out = Path(F_DATA) / "poses" / "vipe"
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=np.float64)
    np.savez_compressed(out / f"{seq}.npz", c2w=poses)
    f_vol.commit()
    return {"seq": seq, "method": "vipe", "n_poses": int(len(poses)),
            "finite": bool(np.isfinite(poses).all()),
            "path_len_units": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                   axis=1).sum()) if len(poses) > 1 else 0.0,
            **extra}


@app.function(image=vipe_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def vipe_probe(seq: str) -> str:
    import subprocess
    import time
    from pathlib import Path

    frames = _stage(seq)
    out = Path("/tmp/vipe_out")
    t0 = time.time()
    cmd = ["vipe", "infer", "--image-dir", frames, "-o", str(out), "-p", "default"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"seq": seq, "cmd": " ".join(cmd), "returncode": r.returncode,
           "wall_s": round(time.time() - t0, 1), "stdout_tail": r.stdout[-4000:],
           "stderr_tail": r.stderr[-4000:]}
    rec["files"] = ([str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()][:100]
                    if out.exists() else [])
    if out.exists():
        import numpy as np
        for path in sorted(out.rglob("*.npz"))[:6]:
            try:
                z = np.load(path)
                rec.setdefault("arrays", {})[str(path)] = {
                    k: list(np.asarray(z[k]).shape) for k in z.files}
            except Exception as exc:  # noqa: BLE001
                rec.setdefault("arrays", {})[str(path)] = f"unreadable: {exc}"
    return json.dumps(rec, indent=1)


@app.function(image=vipe_image, gpu="A100-80GB", volumes={C_DATA: c_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def vipe_run(seqs: list[str]) -> str:
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    recs = []
    for seq in seqs:
        frames = _stage(seq)
        out = Path("/tmp/vipe_out") / seq
        t0 = time.time()
        cmd = ["vipe", "infer", "--image-dir", frames, "-o", str(out), "-p", "default"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rec = {"seq": seq, "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
               "n_frames": 150, "stderr_tail": r.stderr[-1500:]}
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
        rec.update(_save(c2w, seq, {"pose_key": key, "raw_shape": list(arr.shape)}))
        recs.append(rec)
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(seqs: str = "", probe: str = ""):
    chosen = [s for s in seqs.split(",") if s] or SEQS
    if probe:
        print(vipe_probe.remote(probe))
        return
    print(vipe_run.remote(chosen))
