"""Stage the Part 1 defect renders as reconstruction units (Track V).

Reads ``data/v/renders/<edit_id>/{gt.npz,meta.json,<edit_id>.mp4}`` (pulled from the volume
``awmv-v-data:v_render``), decodes every 4th frame (30 fps -> 7.5 Hz), and writes a unit whose
truth is the *commanded* 250 Hz altitude (unchanged by the edit) with the rendered altitude kept
alongside so the predicted effect can be computed exactly.

Usage
-----
PYTHONPATH=. python experiments/v_stage_defects.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v"
FRAMES = OUT / "frames"
UNITS = OUT / "units.json"
RENDERS = OUT / "renders"
SIM_K = {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 179.5, "w": 640, "h": 360}
STRIDE = 4
FPS = 30.0


def stage(edit_id: str) -> dict:
    d = RENDERS / edit_id
    meta = json.loads((d / "meta.json").read_text())
    gt = np.load(d / "gt.npz")
    mp4 = d / f"{edit_id}.mp4"
    dst = FRAMES / "defect" / edit_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
                    "-vf", f"select='not(mod(n\\,{STRIDE}))'", "-fps_mode", "passthrough",
                    "-start_number", "0", str(dst / "%04d.png")], check=True)
    files = sorted(dst.glob("*.png"))
    t = np.arange(len(files)) * STRIDE / FPS
    tt = np.asarray(gt["t"], float)
    pos = np.asarray(gt["pos_commanded"], float)
    rend = np.asarray(gt["pos"], float)
    m = (tt >= t[0] - 1e-6) & (tt <= t[-1] + 1e-6)
    return {"unit": edit_id, "kind": "defect", "edit_family": meta["kind"],
            "clip_id": meta["clip_id"], "uptilt_deg": meta.get("uptilt_deg"),
            "family": meta.get("family"),
            "frames": str(dst.relative_to(FRAMES)), "n_frames": len(files), "K": SIM_K,
            "fps": 7.5,
            "truth": {"t": tt[m].tolist(), "pos": pos[m].tolist(),
                      "rendered_pos": rend[m].tolist(), "ground_z": 0.0,
                      "reversed_factor": meta.get("reversed_factor"),
                      "z0_m": meta.get("z0_m"),
                      "z_rendered_max": meta.get("z_rendered_max")},
            "source": {"volume": "awmv-v-data", "path": f"v_render/{edit_id}"}}


def main() -> int:
    edits = sorted(p.name for p in RENDERS.iterdir() if (p / "meta.json").exists())
    new = []
    for e in edits:
        u = stage(e)
        print(f"{e}: kind={u['edit_family']} frames={u['n_frames']}")
        new.append(u)
    old = json.loads(UNITS.read_text())["units"] if UNITS.exists() else []
    seen = {u["unit"] for u in new}
    merged = [u for u in old if u["unit"] not in seen] + new
    UNITS.write_text(json.dumps({"generated": "experiments/v_stage_defects.py",
                                 "units": merged}, indent=1))
    print(f"{len(new)} defect units staged ({len(merged)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
