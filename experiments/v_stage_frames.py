"""Stage the Track V pilot frames at 7.5 Hz (Track V, Part 0).

Writes ``data/v/frames/<unit>/<frame>.png`` plus ``data/v/units.json``. The unit table is the
provenance: for every unit, the source file, the source frame indices kept, their times, the
intrinsics, and the truth series the analysis reads.

Populations are exactly those of ``configs/v00_feasibility_prereg.yaml``:
  sim  : every dev climb_turn clip + i02_dev_000 (hover control) + i02_dev_001 (ellipse control)
  uzh  : outdoor_forward_1 / _3 / _5, every 2nd prepared frame (15.07 Hz -> 7.5 Hz)

Usage
-----
PYTHONPATH=. python experiments/v_stage_frames.py [--sim-only|--uzh-only]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v"
FRAMES = OUT / "frames"
UNITS = OUT / "units.json"

SIM_CONTROLS = ("i02_dev_000", "i02_dev_001")
UZH_PILOT = ("outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5")
UZH_REST = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
            "indoor_forward_9", "indoor_forward_10", "indoor_45_2", "indoor_45_4",
            "indoor_45_9", "indoor_45_13", "outdoor_45_1")
UZH_ALL = UZH_PILOT + UZH_REST
SIM_K = {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 179.5, "w": 640, "h": 360}
FPS_SIM = 30.0
SIM_STRIDE = 4          # 30 fps -> 7.5 Hz


def sim_pilot_ids() -> list[str]:
    import csv

    rows = list(csv.DictReader(open(ROOT / "results/i02/manifest.csv")))
    dev = [r for r in rows if r["split"] == "dev"]
    climb = sorted(r["clip_id"] for r in dev if r["maneuver"] == "climb_turn")
    have = [c for c in climb if (ROOT / f"data/n05/gt/{c}.npz").exists()]
    if len(have) != len(climb):
        print(f"WARNING: {len(climb) - len(have)} climb clips have no local gt")
    return have + list(SIM_CONTROLS)


def _video_frames(clip_id: str) -> Path:
    return ROOT / f"data/n05/videos/{clip_id}.mp4"


def stage_sim(clip_id: str) -> dict:
    dst = FRAMES / "sim" / clip_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    src = _video_frames(clip_id)
    if not src.exists():
        raise FileNotFoundError(src)
    # select every 4th frame and renumber from 0; setpts keeps the output index == input index
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
           "-vf", f"select='not(mod(n\\,{SIM_STRIDE}))'", "-fps_mode", "passthrough",
           "-start_number", "0", str(dst / "%04d.png")]
    subprocess.run(cmd, check=True)
    files = sorted(dst.glob("*.png"))
    src_idx = np.arange(len(files)) * SIM_STRIDE
    t = src_idx / FPS_SIM
    gt = np.load(ROOT / f"data/n05/gt/{clip_id}.npz")
    # truth restricted to the same time span, at the 250 Hz gt rate
    tt = np.asarray(gt["t"], dtype=float)
    pos = np.asarray(gt["pos"], dtype=float)
    m = (tt >= t[0] - 1e-6) & (tt <= t[-1] + 1e-6)
    return {"unit": clip_id, "kind": "sim", "frames": str(dst.relative_to(FRAMES)),
            "n_frames": len(files), "src": str(src.relative_to(ROOT)),
            "src_frame_index": src_idx.tolist(), "t": t.tolist(),
            "K": SIM_K, "fps": 7.5,
            "truth": {"t": tt[m].tolist(), "pos": pos[m].tolist(),
                      "ground_z": 0.0, "clip_z0": float(pos[m][0, 2])}}


def stage_uzh(seq: str, population: str = "pilot") -> dict:
    src_dir = ROOT / f"data/n01/uzh_frames/{seq}"
    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    files = sorted(src_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(src_dir)
    kept = files[::2]                       # 15.07 Hz -> 7.5 Hz
    dst = FRAMES / "uzh" / seq
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for k, p in enumerate(kept):
        (dst / f"{k:04d}.png").symlink_to(p.resolve())
    fps = float(meta["fps_measured"])
    idx = np.arange(len(kept)) * 2
    t = idx / fps
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    tp = np.asarray(truth["t"], dtype=float)
    pos = np.asarray(truth["pos"], dtype=float)
    m = (tp >= t[0] - 1e-6) & (tp <= t[-1] + 1e-6)
    K = meta["intrinsics_pinhole"]
    return {"unit": seq, "kind": "uzh", "population": population,
            "frames": str(dst.relative_to(FRAMES)),
            "n_frames": len(kept), "src": str(src_dir.relative_to(ROOT)),
            "src_frame_index": idx.tolist(), "t": t.tolist(),
            "K": {"fx": float(K["fx"]), "fy": float(K["fy"]), "cx": float(K["cx"]),
                  "cy": float(K["cy"]), "w": int(K["w"]), "h": int(K["h"])},
            "fps": 7.5,
            "truth": {"t": tp[m].tolist(), "pos": pos[m].tolist(), "ground_z": None,
                      "clip_z0": None}, "meta": {"fps_measured": fps}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--uzh-only", action="store_true")
    ap.add_argument("--uzh-all", action="store_true",
                    help="stage the 11 further UZH sequences as well (V1 real population)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    units = []
    if not args.uzh_only:
        for cid in sim_pilot_ids():
            u = stage_sim(cid)
            print(f"sim {cid}: {u['n_frames']} frames, t {u['t'][0]:.2f}-{u['t'][-1]:.2f}")
            units.append(u)
    if not args.sim_only:
        seqs = UZH_ALL if args.uzh_all else UZH_PILOT
        for seq in seqs:
            u = stage_uzh(seq, "pilot" if seq in UZH_PILOT else "v1_real")
            print(f"uzh {seq}: {u['n_frames']} frames, t {u['t'][0]:.2f}-{u['t'][-1]:.2f}")
            units.append(u)
    old = []
    if UNITS.exists():
        old = [u for u in json.loads(UNITS.read_text())["units"]]
    seen = {u["unit"] for u in units}
    merged = units + [u for u in old if u["unit"] not in seen]
    UNITS.write_text(json.dumps({"generated": "experiments/v_stage_frames.py",
                                 "populations": {"sim": "28 climb_turn dev + 2 controls",
                                                 "uzh_pilot": list(UZH_PILOT),
                                                 "uzh_all": list(UZH_ALL)},
                                 "units": merged}, indent=1))
    print(f"\n{len(merged)} units -> {UNITS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
