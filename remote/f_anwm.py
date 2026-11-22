"""Track F, stage 2: run the stage-1 extractor on the ANWM clips (generated video).

The ANWM clips have a **known commanded path**, so path recovery can be scored without
Leica truth. The frames live on the ``awmv-b-results`` volume
(``clips/<A|B|C>/<clip>/frames`` and ``clips_injected/injected_true/<A|B|C>/<clip>/frames``,
16 frames per clip); the commanded paths are the committed specs under
``data/processed/anwm_specs/clips_v1``.

This app only extracts. Scoring is CPU-side in ``experiments/f03_anwm.py``.

    pixi run modal run remote/f_anwm.py --list
    pixi run modal run remote/f_anwm.py --kind level --limit 4
    pixi run modal run remote/f_anwm.py --kind level
"""
from __future__ import annotations

import json

import modal

app = modal.App("awmv-f-anwm")

b_vol = modal.Volume.from_name("awmv-b-results", create_if_missing=True)
f_vol = modal.Volume.from_name("awmv-f-data", create_if_missing=True)
B_DATA = "/b-results"
F_DATA = "/f-data"

vipe_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg", "libopenexr-dev",
                 "build-essential", "clang")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy<2.3", "wheel", "pip>=24.2", "setuptools>=77", "ninja", "packaging")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "CUDA_HOME": "/usr/local/cuda"})
    .run_commands("MAX_JOBS=8 pip install --no-build-isolation nvidia-vipe==1.2.0")
    .env({"HF_HOME": f"{F_DATA}/hf", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
)


def _clip_dirs(kind: str) -> list[str]:
    """Every clip directory that carries a ``frames`` subdirectory."""
    from pathlib import Path

    root = Path(B_DATA) / ("clips" if kind == "level" else "clips_injected/injected_true")
    out = []
    for frames in sorted(root.rglob("frames")):
        if any(frames.glob("*.png")):
            out.append(str(frames.parent.relative_to(B_DATA)))
    return out


@app.function(image=vipe_image, volumes={B_DATA: b_vol, F_DATA: f_vol},
              timeout=600, cpu=2)
def list_clips() -> str:
    return json.dumps({"level": len(_clip_dirs("level")),
                       "injected": len(_clip_dirs("injected"))}, indent=1)


@app.function(image=vipe_image, gpu="A100-80GB", volumes={B_DATA: b_vol, F_DATA: f_vol},
              timeout=3600, cpu=8)
def anwm_vipe(kind: str, limit: int = 0, only: str = "") -> str:
    import shutil
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    clips = _clip_dirs(kind)
    if only:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        clips = [c for c in clips if Path(c).name in keep]
    if limit:
        clips = clips[:limit]
    recs = []
    for rel in clips:
        src = Path(B_DATA) / rel / "frames"
        local = Path("/tmp") / Path(rel).name / "frames"
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        for p in sorted(src.glob("*.png")):
            shutil.copy(p, local / p.name)
        out = Path("/tmp/vipe_out") / Path(rel).name
        t0 = time.time()
        cmd = ["vipe", "infer", "--image-dir", str(local), "-o", str(out), "-p", "default"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rec = {"clip": rel, "kind": kind, "returncode": r.returncode,
               "wall_s": round(time.time() - t0, 1), "n_frames_in": len(list(local.glob("*.png"))),
               "stderr_tail": r.stderr[-800:]}
        npzs = sorted((out / "pose").glob("*.npz")) if out.exists() else []
        if r.returncode != 0 or not npzs:
            rec["error"] = "run failed" if r.returncode else "no pose npz"
            recs.append(rec)
            continue
        z = np.load(npzs[0])
        if "data" in z.files and "inds" in z.files:
            inds = np.asarray(z["inds"]).astype(int)
            arr = np.asarray(z["data"], dtype=np.float64)[np.argsort(inds)]
        else:
            key = "pose" if "pose" in z.files else ("c2w" if "c2w" in z.files else z.files[0])
            arr = np.asarray(z[key], dtype=np.float64)
        if arr.shape[-2:] == (3, 4):
            c2w = np.tile(np.eye(4), (arr.shape[0], 1, 1))
            c2w[:, :3, :4] = arr
        elif arr.shape[-2:] == (4, 4):
            c2w = arr
        else:
            rec["error"] = f"pose shape {arr.shape}"
            recs.append(rec)
            continue
        dst = Path(F_DATA) / "poses_anwm" / "vipe" / (rel.replace("/", "__") + ".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        commanded = None
        man = Path(B_DATA) / rel / "manifest.json"
        if man.exists():
            try:
                doc = json.loads(man.read_text())
                pos = doc.get("commanded_path", {}).get("positions")
                if pos is not None:
                    commanded = np.asarray(pos, dtype=np.float64)
            except Exception:  # noqa: BLE001
                commanded = None
        if commanded is not None:
            np.savez_compressed(dst, c2w=c2w, commanded=commanded)
        else:
            np.savez_compressed(dst, c2w=c2w)
        f_vol.commit()
        rec.update({"n_poses": int(len(c2w)), "finite": bool(np.isfinite(c2w).all()),
                    "has_commanded": commanded is not None,
                    "path_len_units": float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0),
                                                           axis=1).sum())
                    if len(c2w) > 1 else 0.0})
        recs.append(rec)
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(kind: str = "level", limit: int = 0, only: str = "", list_only: bool = False):
    if list_only:
        print(list_clips.remote())
        return
    print(anwm_vipe.remote(kind, limit, only))
