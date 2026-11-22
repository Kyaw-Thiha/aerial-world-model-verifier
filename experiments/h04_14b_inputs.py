"""H04 amendment 3: assemble the 35-sample 14B pilot input set.

20 arm-A continuous-attitude clips (draw 0 of every path with a bank profile) + the 15
held-bank flatness clips from H02's grid (turn5/10/20 x five first frames).

    PYTHONPATH=. pixi run -e gpu python experiments/h04_14b_inputs.py

Writes ``data/h04_14b_inputs/<sid>/{image.png,poses.npy,prompt.txt,meta.json}`` and a
manifest; images are resized to 480x832 (the 14B release's resolution).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/h04_solarwm_inputs"
H02 = Path("/home/kevin/Documents/Projects/aerial-world-model-verifier/data/h02_solarwm_inputs")
OUT = ROOT / "data/h04_14b_inputs"
W, H = 832, 480


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    samples = []
    # 20 arm-A draw-0 clips: every path except the two pitch-only accel_brake paths
    summary = json.loads((ROOT / "results/h04/paths/paths_summary.json").read_text())
    paths = [s["path"] for s in summary if s["family"] != "accel_brake"]
    assert len(paths) == 20, len(paths)
    for name in paths:
        sid = f"a_{name}"
        src = SRC / f"main_{name}_A_flatness_g0"
        d = OUT / sid
        d.mkdir(exist_ok=True)
        Image.open(src / "image.png").convert("RGB").resize((W, H), Image.BICUBIC).save(
            d / "image.png")
        np.save(d / "poses.npy", np.load(src / "poses.npy").astype(np.float32))
        shutil.copyfile(src / "prompt.txt", d / "prompt.txt")
        meta = json.loads((src / "meta.json").read_text())
        meta.update({"kind": "armA_continuous", "path": name, "arm": "A_flatness",
                     "draw": 0, "source_sample": f"main_{name}_A_flatness_g0"})
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        samples.append(sid)
    # 15 held-bank clips from H02's grid
    held = sorted(p for p in H02.glob("grid_*_turn*_flatness") if p.is_dir())
    assert len(held) == 15, len(held)
    for p in held:
        sid = "b_" + p.name
        d = OUT / sid
        d.mkdir(exist_ok=True)
        Image.open(p / "image.png").convert("RGB").resize((W, H), Image.BICUBIC).save(
            d / "image.png")
        np.save(d / "poses.npy", np.load(p / "poses.npy").astype(np.float32))
        shutil.copyfile(p / "prompt.txt", d / "prompt.txt")
        src_meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        meta = {"kind": "held_bank", "h02_sample": p.name,
                "path": src_meta.get("path"), "arm": "flatness",
                "source_clip": src_meta.get("source_clip"),
                "traj": src_meta.get("traj"), "frame": src_meta.get("frame")}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        samples.append(sid)
    (OUT / "manifest.json").write_text(json.dumps({"samples": samples, "n": len(samples),
                                                   "size": [W, H]}, indent=1))
    print(json.dumps({"n": len(samples), "armA": sum(s.startswith("a_") for s in samples),
                      "held": sum(s.startswith("b_") for s in samples)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
