"""X01: the frozen depicted-attitude reader on the common-path-set clips.

Same feed as H03/H04/G06: square centre crop, 224x224, GeoCalib pinhole at commit
``1dedb29`` with a scalar focal prior f = 112 px (90 deg FOV).  Reads
``/r/x01/<model>/<tag>/<sample>/frames/*.png`` from the ``awmv-x-results`` volume and writes
one JSONL of per-frame rows per (model, tag, shard).

    set -a; . ./.env; set +a
    pixi run modal run --detach remote/x01_readers.py::main --model solarwm --tag d42
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from src.remote_common import GEOCALIB_COMMIT

vol = modal.Volume.from_name("awmv-x-results", create_if_missing=True)

app = modal.App("awmv-x-readers")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib", "kornia", "gdown",
                 "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)


@app.function(image=image, gpu="A10", timeout=4 * 3600, volumes={"/r": vol},
              env={"HF_HOME": "/tmp/hf"})
def depicted(model: str = "solarwm", tag: str = "d42", size: int = 224,
             fov_deg: float = 90.0, stride: int = 1, shard: int = 0,
             shards: int = 1) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.remote_common import gate_row

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib(weights="pinhole").to(device)
    root = Path("/r/x01") / model / tag
    clips = sorted(p for p in root.iterdir() if (p / "frames").is_dir())[shard::shards]
    out_dir = Path("/r/x01") / "depicted" / f"{model}_{tag}_s{shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    t0 = time.time()
    n = 0
    with (out_dir / "depicted.jsonl").open("w") as fh:
        for clip in clips:
            frames = sorted((clip / "frames").glob("*.png"))[::max(1, int(stride))]
            for p in frames:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                orig = [int(img.shape[1]), int(img.shape[0])]
                h, w = img.shape[:2]
                if h != w:
                    side = min(h, w)
                    y0, x0 = (h - side) // 2, (w - side) // 2
                    img = img[y0:y0 + side, x0:x0 + side]
                if img.shape[0] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                feed = feed.float().to(device) / 255.0
                with torch.no_grad():
                    res = gc.calibrate(feed, camera_model="pinhole",
                                       priors={"focal": torch.tensor(float(f), device=device)})
                row = gate_row(res, {"frame": p.name, "clip_id": clip.name,
                                     "frame_std": float(img.std()), "orig_size": orig,
                                     "model": model, "tag": tag})
                fh.write(json.dumps(row) + "\n")
                n += 1
            fh.flush()
            vol.commit()
            print(clip.name, len(frames), n, flush=True)
    info = {"model": model, "tag": tag, "shard": shard, "shards": shards,
            "n_clips": len(clips), "n_rows": n, "size": size, "f_px": float(f),
            "stride": stride, "seconds": time.time() - t0,
            "geocalib_commit": GEOCALIB_COMMIT}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
    vol.commit()
    return info


@app.local_entrypoint()
def main(model: str = "solarwm", tag: str = "d42", shard: int = 0, shards: int = 1,
         stride: int = 1):
    calls = [depicted.spawn(model=model, tag=tag, shard=k, shards=shards, stride=stride)
             for k in range(shards)]
    print(json.dumps([c.get() for c in calls], indent=1))


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes={"/r": vol})
def collect(prefixes: str = "depicted,solarwm,anwm,lingbot,sana") -> dict:
    """Return every small artefact (JSON/JSONL/MD, not png/mp4) under the given subtrees."""
    root = Path("/r/x01")
    out: dict[str, str] = {}
    for sub in prefixes.split(","):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".log", ".txt"):
                if p.stat().st_size > 8_000_000:
                    continue
                out[str(p.relative_to(root))] = p.read_text(errors="replace")
    return out


PANEL = ["arc_10", "arc_20", "s_10", "climb_02", "accel_brake_10", "arc_25", "climb_turn_20",
         "real_uzh_of1", "arc_35", "climb_turn_30", "real_uzh_of3"]


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={"/r": vol})
def sheets(max_clips: int = 40, per_clip: int = 6) -> dict:
    """Side-by-side contact sheets for the human-judgement check (Stage 3c).

    One sheet per clip: ``per_clip`` frames evenly spaced over the clip, each labelled with its
    frame index, and a header carrying the generator, the path and the peak tilt the path
    requires.  No verdict is drawn on the sheet: the reader is asked for a judgement, not shown
    one.  A labels CSV is written beside the sheets.
    """
    import cv2
    import numpy as np

    root = Path("/r/x01")
    out = root / "human"
    out.mkdir(parents=True, exist_ok=True)
    groups = [("solarwm", "d42"), ("solarwm", "d43"), ("anwm", "C"), ("sana", "full"),
              ("lingbot", "x01")]
    rows = []
    for model, tag in groups:
        base = root / model / tag
        if not base.exists():
            continue
        if model == "anwm":
            base = base / "C"
        for d in sorted(p for p in base.glob("*") if (p / "frames").is_dir()):
            sid = d.name
            path = sid[4:].rsplit("_", 1)[0] if sid.startswith("x01_") else \
                (sid[:-4] if sid.endswith("__S1") else sid)
            if path not in PANEL and sid not in PANEL:
                continue
            if model == "solarwm" and tag == "d42":
                pass
            elif model == "solarwm" and tag == "d43":
                pass
            elif "S2" in sid:
                pass
            rows.append((model, tag, sid, path, d))
    rows = rows[:max_clips]
    labels = []
    for k, (model, tag, sid, path, d) in enumerate(rows):
        frames = sorted((d / "frames").glob("*.png"))
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, per_clip).round().astype(int)
        tiles = []
        for i in idx:
            im = cv2.imread(str(frames[i]))
            if im is None:
                continue
            h, w = im.shape[:2]
            s = 320.0 / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)))
            cv2.putText(im, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)
            tiles.append(im)
        if not tiles:
            continue
        hh = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 2, cv2.BORDER_CONSTANT,
                                    value=(40, 40, 40)) for t in tiles]
        sheet = np.concatenate(tiles, axis=1)
        head = np.full((46, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{model}/{tag}  {path}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1)
        cv2.putText(head, f"clip {sid}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        sheet = np.concatenate([head, sheet], axis=0)
        name = f"{k:02d}_{model}_{path}".replace("/", "_")
        cv2.imwrite(str(out / f"{name}.png"), sheet)
        labels.append({"sheet": f"{name}.png", "model": model, "tag": tag, "clip_id": sid,
                       "path": path})
    (out / "labels.csv").write_text(
        "sheet,model,draw,clip_id,path,plausible_1_to_5,comment\n"
        + "\n".join(f"{r['sheet']},{r['model']},{r['tag']},{r['clip_id']},{r['path']},," for r in labels)
        + "\n")
    vol.commit()
    return {"n_sheets": len(labels), "out": str(out), "labels": labels}
