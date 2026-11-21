"""Track H2 Part 2: SolarWM-Wan2.2-5B Stage2 camera-controlled inference on Modal.

Weights: ``junchaoh-cs/SolarWM-Wan2.2-5B`` (gated; the account has accepted the terms).
Code and environment follow the release-tested Wan route: Python 3.10, torch 2.5.1/cu124,
diffusers 0.38.0, transformers 5.12.1, flash-attn 2.8.3.

The app builds a minimal raw-WDS sample set from ``data/h02_solarwm_inputs`` (image, 81-frame
OpenCV c2w, prompt), writes a local index, and runs the released ``solar infer`` Stage2
camera-length pipeline. Outputs are decoded to PNG frames for the existing reader.

    set -a; . ./.env; set +a
    pixi run modal run remote/h02_solarwm.py::main --mode download
    pixi run modal run remote/h02_solarwm.py::main --mode test
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h02_solarwm_inputs"
CODE = Path("/tmp/SolarWM")

vol_w = modal.Volume.from_name("awmv-h2-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-h2-results", create_if_missing=True)
DEST = "/w/solarwm"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg", "ninja-build")
    .run_commands(
        "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("numpy<2", "scipy", "pandas", "pillow", "opencv-python-headless",
                 "decord", "einops", "safetensors", "ftfy", "omegaconf", "regex",
                 "pyyaml", "tqdm", "imageio", "imageio-ffmpeg", "av",
                 "diffusers==0.38.0", "transformers==5.12.1", "peft==0.20.0",
                 "huggingface_hub")
    .run_commands(
        "pip install --no-cache-dir "
        "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' "
        "|| echo 'flash-attn wheel unavailable'")
    .add_local_dir(str(CODE), "/solarwm", copy=True,
                   ignore=[".git", "**/.git", "**/__pycache__", "**/*.pyc"])
    .run_commands("cd /solarwm && pip install --no-cache-dir --no-deps -e .")
    .add_local_dir(str(INPUTS), "/inputs", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("awmv-h2-solarwm", image=image)


@app.function(cpu=16, memory=32768, timeout=4 * 3600, volumes={"/w": vol_w},
              env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
def download() -> dict:
    import os

    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]
    snapshot_download("junchaoh-cs/SolarWM-Wan2.2-5B", local_dir=DEST, token=token,
                      allow_patterns=["config.json", "SolarWM-5B-base/**",
                                      "SolarWM-5B-sgf-stage2-81f/**"])
    vol_w.commit()
    files = sorted(p for p in Path(DEST).rglob("*") if p.is_file())
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "biggest": sorted(((p.stat().st_size, str(p.relative_to(DEST)))
                               for p in files), reverse=True)[:6]}


def _build_data(sample_ids: list[str], out_root: Path) -> Path:
    import gzip
    import json as _json
    import shutil

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    release = out_root / "releases-v1"
    raw_dir = release / "raw" / "h02"
    recipe_dir = release / "recipes" / "h02"
    raw_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)
    shard = raw_dir / "shard.tar"
    rows = []
    import tarfile

    with tarfile.open(shard, "w") as tar:
        for sid in sample_ids:
            src = Path("/inputs") / sid
            frames = [Image.open(src / "image.png").convert("RGB")] * 81
            mp4 = raw_dir / f"{sid}.mp4"
            imageio.mimsave(mp4, [np.asarray(f) for f in frames], fps=16,
                            macro_block_size=1)
            poses = np.load(src / "poses.npy").astype("float32")
            npz = raw_dir / f"{sid}.camera.npz"
            np.savez(npz, c2w=poses)
            prompt = (src / "prompt.txt").read_text().strip()
            manifest = {
                "video": {"num_frames": 81, "fps": 16.0, "height": 480, "width": 864},
                "camera": {
                    "array_key": "c2w",
                    "convention": "authoritative_source_c2w_no_axis_flip",
                    "dtype": "float32", "finite": True,
                    "magnitude_audit_seconds": 10.0,
                    "max_camera_abs": 20.0, "max_rel_translation": 20.0,
                    "shape": [81, 4, 4],
                },
                "prompt": {"text": prompt},
                "metadata": {"scene": "h02", "dataset": "h02"},
            }
            for path, arc in ((mp4, f"{sid}.mp4"), (npz, f"{sid}.camera.npz")):
                tar.add(path, arcname=arc)
            rows.append({
                "sample_id": sid, "key": sid, "shard": "raw/h02/shard.tar",
                "fps": 16.0, "num_frames": 81, "start_frame": 0,
                "video_member": f"{sid}.mp4", "camera_member": f"{sid}.camera.npz",
                "manifest": manifest,
            })
            shutil.rmtree(raw_dir / "tmp", ignore_errors=True)
    with gzip.open(recipe_dir / "test-index.jsonl.gz", "wt") as fh:
        for row in rows:
            fh.write(_json.dumps(row) + "\n")
    return release


def _write_config(release: Path, out: Path, count: int, steps: int,
                  frame_num: int) -> Path:
    import yaml

    template = Path("/solarwm/configs/solar_wm/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    cfg = yaml.safe_load(template.read_text())
    cfg["model"]["base_path"] = f"{DEST}/SolarWM-5B-base"
    cfg["checkpoint"]["path"] = f"{DEST}/SolarWM-5B-sgf-stage2-81f"
    cfg["checkpoint"]["weights"] = ""
    cfg["data"]["index_root"] = str(release)
    cfg["data"]["transport"] = {"kind": "local", "root": str(release)}
    cfg["data"]["test_index"] = "recipes/h02/test-index.jsonl.gz"
    cfg["data"]["pixel_frames"] = frame_num
    cfg["data"]["latent_frames"] = 1 + (frame_num - 1) // 4
    cfg["data"]["max_rel_translation"] = 20.0
    cfg["data"]["max_camera_abs"] = 20.0
    cfg["validation"]["sample_count"] = count
    cfg["validation"]["noise_seed"] = 42
    cfg["validation"]["selection_seed"] = 42
    cfg["validation"]["passes"] = [{
        "name": "model_self_forcing_nfe4", "weights": "model",
        "mode": "autoregressive", "solver": "self_forcing",
        "num_inference_steps": steps,
        "rollout_latent_frames": cfg["data"]["latent_frames"],
    }]
    cfg["inference"] = {"source": "validation", "length": "camera",
                        "output_layout": "transaction_v1",
                        "run_id": f"h02-{out.name}"}
    cfg["runtime"]["output_dir"] = str(out)
    path = Path("/tmp") / f"{out.name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@app.function(gpu="A100-80GB", cpu=8, memory=65536, timeout=6 * 3600,
              volumes={"/w": vol_w, "/r": vol_r}, env={"HF_HOME": "/w/hf"})
def infer(sample_ids: str, tag: str, steps: int = 4, frame_num: int = 81) -> dict:
    import os
    import subprocess
    import time

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    ids = json.loads(sample_ids)
    out_root = Path("/r/solarwm") / tag
    out_root.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp") / f"data_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    release = _build_data(ids, work)
    out = out_root / "run"
    cfg = _write_config(release, out, len(ids), steps, frame_num)
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH="/solarwm")
    proc = subprocess.run(["python", "-m", "solar", "infer", "--config", str(cfg)],
                          cwd="/solarwm", env=env, capture_output=True, text=True)
    (out_root / "infer.log").write_text(proc.stdout[-200000:] + "\n---STDERR---\n"
                                       + proc.stderr[-100000:])
    if proc.returncode != 0:
        raise RuntimeError(f"solar infer failed: {proc.returncode}; see infer.log")
    # decode generated mp4s to PNG frames
    clips = {}
    for mp4 in sorted(out.rglob("*.mp4")):
        if "compare" in mp4.parts:
            continue
        sid = mp4.stem
        frames = list(imageio.get_reader(mp4))
        d = out_root / sid / "frames"
        d.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            Image.fromarray(np.asarray(fr)).save(d / f"{i:03d}.png")
        clips[sid] = {"frames": len(frames), "mp4": str(mp4.relative_to(out_root))}
    vol_r.commit()
    return {"tag": tag, "n_samples": len(ids), "seconds": time.time() - t0,
            "returncode": proc.returncode, "clips": clips,
            "output_root": str(out_root)}


@app.local_entrypoint()
def main(mode: str = "download", tag: str = "test", ids: str = "", steps: int = 4):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    man = json.loads((INPUTS / "manifest.json").read_text())
    if not ids:
        chosen = ["probe_p20"] if mode == "test" else man["samples"]
    else:
        chosen = ids.split(",")
    print(f"{len(chosen)} samples, tag {tag}")
    res = infer.remote(json.dumps(chosen), tag=tag, steps=steps)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(json.dumps(res["clips"], indent=1)[:3000])
