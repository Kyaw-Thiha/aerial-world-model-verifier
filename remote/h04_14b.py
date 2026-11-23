"""H04 amendment 3: the SolarWM 14B pilot.

Release used (inspected before download):
``junchaoh-cs/SolarWM-Wan2.2-14B`` = ``SolarWM-14B-base-high`` (Wan2.2 I2V-A14B high-noise
expert + T5) + ``SolarWM-14B-bid-stage0p5-81f`` (114 GB consolidated stage0.5 checkpoint).
Stage0.5 flow matching, 50 UniPC steps, bidirectional, 480x832 — **not** the 5B's Stage2
4-step causal route.

    set -a; . ./.env; set +a
    pixi run modal run remote/h04_14b.py::main --mode download
    pixi run modal run remote/h04_14b.py::main --mode timing --ids a,b,c
    pixi run modal run remote/h04_14b.py::main --mode read --ids ... --tag pilot14b
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h04_14b_inputs"
CODE = Path("/tmp/SolarWM")
DEST = "/w/14b"

vol_w = modal.Volume.from_name("awmv-h3-14b-weights", create_if_missing=True)
vol_r = modal.Volume.from_name("awmv-h3-results", create_if_missing=True)

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

app = modal.App("awmv-h3-solarwm14b", image=image)


@app.function(cpu=16, memory=65536, timeout=6 * 3600, volumes={"/w": vol_w},
              env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
def download() -> dict:
    from huggingface_hub import snapshot_download

    snapshot_download("junchaoh-cs/SolarWM-Wan2.2-14B", local_dir=DEST,
                      token=os.environ["HF_TOKEN"],
                      allow_patterns=["config.json", "SolarWM-14B-base-high/**",
                                      "SolarWM-14B-bid-stage0p5-81f/**"])
    vol_w.commit()
    files = [p for p in Path(DEST).rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def _build_data(sample_ids: list[str], out_root: Path, width: int = 832, height: int = 480,
                frames: int = 81) -> Path:
    import gzip
    import json as _json
    import shutil
    import tarfile

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    release = out_root / "releases-v1"
    raw_dir = release / "raw" / "h04b"
    recipe_dir = release / "recipes" / "h04b"
    raw_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tarfile.open(raw_dir / "shard.tar", "w") as tar:
        for sid in sample_ids:
            src = Path("/inputs") / sid
            im = Image.open(src / "image.png").convert("RGB").resize((width, height),
                                                                     Image.BICUBIC)
            mp4 = raw_dir / f"{sid}.mp4"
            imageio.mimsave(mp4, [np.asarray(im)] * frames, fps=16, macro_block_size=1)
            poses = np.load(src / "poses.npy").astype("float32")
            npz = raw_dir / f"{sid}.camera.npz"
            np.savez(npz, c2w=poses)
            prompt = (src / "prompt.txt").read_text().strip()
            manifest = {
                "video": {"num_frames": frames, "fps": 16.0, "height": height,
                          "width": width},
                "camera": {"array_key": "c2w",
                           "convention": "authoritative_source_c2w_no_axis_flip",
                           "dtype": "float32", "finite": True,
                           "magnitude_audit_seconds": 10.0,
                           "max_camera_abs": 20.0, "max_rel_translation": 20.0,
                           "shape": [frames, 4, 4]},
                "prompt": {"text": prompt},
                "metadata": {"scene": "h04b", "dataset": "h04b"},
            }
            for path, arc in ((mp4, f"{sid}.mp4"), (npz, f"{sid}.camera.npz")):
                tar.add(path, arcname=arc)
            rows.append({"sample_id": sid, "key": sid, "shard": "raw/h04b/shard.tar",
                         "fps": 16.0, "num_frames": frames, "start_frame": 0,
                         "video_member": f"{sid}.mp4",
                         "camera_member": f"{sid}.camera.npz", "manifest": manifest})
    with gzip.open(recipe_dir / "test-index.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")
    return release


def _write_config(release: Path, out: Path, count: int, steps: int, frames: int,
                  world_size: int = 4, sp: int = 2) -> Path:
    import yaml

    template = Path("/solarwm/configs/solar_wm/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    cfg = yaml.safe_load(template.read_text())
    cfg["model"]["base_path"] = f"{DEST}/SolarWM-14B-base-high"
    cfg["checkpoint"]["path"] = f"{DEST}/SolarWM-14B-bid-stage0p5-81f"
    cfg["checkpoint"]["weights"] = "live"
    cfg["data"]["index_root"] = str(release)
    cfg["data"]["transport"] = {"kind": "local", "root": str(release)}
    cfg["data"]["test_index"] = "recipes/h04b/test-index.jsonl.gz"
    cfg["data"]["pixel_frames"] = frames
    cfg["data"]["latent_frames"] = 1 + (frames - 1) // 4
    cfg["validation"]["sample_count"] = count
    cfg["validation"]["noise_seed"] = 42
    cfg["validation"]["selection_seed"] = 42
    cfg["validation"]["passes"] = [{
        "name": "stage0p5_bidirectional", "weights": "live", "mode": "bidirectional",
        "solver": "unipc", "num_inference_steps": steps,
        "rollout_latent_frames": cfg["data"]["latent_frames"],
    }]
    cfg["inference"] = {"source": "validation"}
    dist = dict(cfg.get("distributed", {}))
    dist.update({"world_size": world_size, "sequence_parallel_size": sp,
                 "sample_sharding": "logical_dp"})
    dist.setdefault("sequence_layout", "whole_frame_uneven")
    cfg["distributed"] = dist
    # training-only fields are still validated at config load: DP*micro*grad_accum must equal
    # global_batch_size, and DP = world_size / sp
    dp = max(1, world_size // max(1, sp))
    cfg.setdefault("train", {})
    cfg["train"]["micro_batch_size"] = 1
    cfg["train"]["grad_accum"] = 1
    cfg["train"]["global_batch_size"] = dp
    cfg["runtime"]["output_dir"] = str(out)
    path = Path("/tmp") / f"{out.name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@app.function(gpu="A100-80GB:4", cpu=8, memory=131072, timeout=6 * 3600,
              volumes={"/w": vol_w, "/r": vol_r}, env={"HF_HOME": "/tmp/hf"})
def infer(sample_ids: str, tag: str, steps: int = 50, frames: int = 81,
          measure: bool = False, world_size: int = 4) -> dict:
    import shutil
    import subprocess
    import time
    import uuid

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    ids = json.loads(sample_ids)
    out_root = Path("/r/solarwm") / tag
    out_root.mkdir(parents=True, exist_ok=True)
    done = [s for s in ids if (out_root / s / "frames").exists()]
    ids = [s for s in ids if s not in done]
    if not ids:
        return {"tag": tag, "clips": 0, "already_done": len(done), "seconds": 0.0,
                "seconds_per_clip": 0.0, "measure": measure}
    work = Path("/tmp") / f"data_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    release = _build_data(ids, work)
    out = Path("/tmp/solar_out") / f"{tag}-{uuid.uuid4().hex[:8]}" / "run"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _write_config(release, out, len(ids), steps, frames, world_size=world_size)
    if measure == "config":
        import yaml as _yaml
        doc = _yaml.safe_load(Path(cfg).read_text())
        return {"config_distributed": doc["distributed"],
                "config_train": {k: doc["train"][k] for k in
                                 ("micro_batch_size", "grad_accum", "global_batch_size")},
                "passes": doc["validation"]["passes"], "ids": len(ids)}
    env = dict(os.environ, PYTHONPATH="/solarwm")
    t0 = time.time()
    proc = subprocess.run(["torchrun", "--standalone",
                           f"--nproc-per-node={world_size}",
                           "-m", "solar", "infer", "--config", str(cfg)],
                          cwd="/solarwm", env=env, capture_output=True, text=True)
    secs = time.time() - t0
    (out_root / "infer.log").write_text(proc.stdout[-200000:] + "\n---STDERR---\n"
                                       + proc.stderr[-100000:])
    vol_r.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"solar infer failed: {proc.returncode}")
    n = 0
    for mf in sorted(out.glob("generation/**/slot-*/manifest.json")):
        doc = json.loads(mf.read_text())
        case = doc.get("case", {}) if isinstance(doc, dict) else {}
        sid = str(doc.get("sample_id") or case.get("sample_id") or case.get("key")
                  or mf.parent.name)
        mp4 = mf.parent / "video.mp4"
        if not mp4.exists():
            continue
        d = out_root / sid / "frames"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mp4, out_root / sid / "clip.mp4")
        shutil.copyfile(mf, out_root / sid / "manifest.json")
        reader = imageio.get_reader(mp4)
        try:
            for i, fr in enumerate(reader):
                Image.fromarray(np.asarray(fr)).save(d / f"{i:03d}.png")
        finally:
            reader.close()
        n += 1
    vol_r.commit()
    return {"tag": tag, "clips": n, "already_done": len(done), "seconds": secs,
            "seconds_per_clip": secs / max(n, 1), "measure": measure}


@app.local_entrypoint()
def main(mode: str = "download", ids: str = "", tag: str = "pilot14b", steps: int = 50):
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
        return
    chosen = ids.split(",") if ids else json.loads(
        (INPUTS / "manifest.json").read_text())["samples"]
    print(f"{len(chosen)} samples, tag {tag}, {steps} steps")
    print(json.dumps(infer.remote(json.dumps(chosen), tag=tag, steps=steps,
                                  measure=(mode == "timing"),
                                  world_size=int(os.environ.get("H04_14B_WORLD", "4"))),
                     indent=1))
