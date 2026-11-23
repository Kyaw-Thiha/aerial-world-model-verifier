"""Track M, Stage 2: stage the Gate B reader inputs and build the frame plan.

Local, CPU, zero GPU.  Pulls the simulated dev clips, the H04 SolarWM clips and the H01 ANWM
frames from R2 into ``data/m02_inputs`` (gitignored), and writes ``plan.json``, which the
Modal app ``remote/m_readers.py`` consumes.  Nothing in another track's prefix is written.

The TII frames are not staged: their camera zips are already on the ``awmv-a-data`` volume,
which the reader app mounts read-only, and the per-frame edit rotations are the stored
``edits/p3a_v1`` specs from Track A2.

    PYTHONPATH=. pixi run python experiments/m02_upload_inputs.py --download
    pixi run modal volume put awmv-m-data data/m02_inputs /m02/inputs --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m02_inputs"
MANIFEST = ROOT / "results/m02/i02_manifest_copy.csv"
SIM_STRIDE = 4          # 30 Hz -> 7.5 Hz
H04_STRIDE = 2
TII_STRIDE = 2
# Focal prior for the 224 px centre crop, per source.  The feed is a square centre crop
# resized to 224; the prior must describe the *content* that survived the crop.
#   TII  : H03's convention for the fisheye centre (a 90 deg pinhole reading of a crop).
#   sim  : the render is 640x360 at 90 deg horizontal, f = 320 px, so a 360 px crop
#          resized to 224 keeps f = 320 * 224/360 = 199.1 px.
#   H04/H01: 224x224 renders at 90 deg, f = 112 px (H03's generated-frame feed).
F_TII, F_SIM, F_GEN = 112.0, 320.0 * 224.0 / 360.0, 112.0
TII_CONDITIONS = {
    "identity": "real",
    "tilt_k0": "tilt_removal",
    "tilt_k05": "scale_0.5",
    "tilt_k15": "scale_1.5",
    "mirror_roll": "mirror_roll",
    "yaw_control": "yaw_control",
}
TII_FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
               "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
               "flight-13p-trackRATM", "flight-16p-trackRATM"]


def sim_plan() -> list[dict]:
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "dev"
            and r["status"] == "ok"]
    out = []
    for r in rows:
        out.append({"source": "sim", "clip_id": r["clip_id"], "family": r["family"],
                    "defect_family": r["defect_family"], "uptilt_deg": float(r["uptilt_deg"]),
                    "maneuver": r["maneuver"], "n_frames": int(r["n_frames"]),
                    "stride": SIM_STRIDE, "f_px": F_SIM,
                    "video": r["r2_video"], "gt": r["r2_gt"],
                    "sha256_video": r["sha256_video"], "sha256_gt": r["sha256_gt"]})
    return out


def tii_plan() -> list[dict]:
    clips = json.loads((ROOT / "data/processed/p3a/p3a_v1/clips.json").read_text())["clips"]
    out = []
    for c in clips:
        out.append({"source": "tii", "clip_id": c["clip_id"], "flight": c["flight"],
                    "control": c["control"], "n_frames": int(c["n_frames"]),
                    "index": [int(v) for v in c["index"]], "stride": TII_STRIDE,
                    "f_px": F_TII, "conditions": list(TII_CONDITIONS),
                    "zip": f"{c['control']}/camera_{c['flight']}.zip"})
    return out


def h04_plan() -> list[dict]:
    from src.utils.r2 import client, load_env
    env = load_env(".env")
    c = client(env)
    pag = c.get_paginator("list_objects_v2")
    out = []
    for page in pag.paginate(Bucket=env["R2_BUCKET"], Prefix="track-h3/clips/solarwm-main/"):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/clip.mp4"):
                continue
            clip = o["Key"].split("/")[3]
            out.append({"source": "h04", "clip_id": clip, "stride": H04_STRIDE,
                        "f_px": F_GEN, "video": o["Key"], "bytes": o["Size"]})
    return sorted(out, key=lambda r: r["clip_id"])


def h01_plan() -> list[dict]:
    from src.utils.r2 import client, load_env
    env = load_env(".env")
    c = client(env)
    pag = c.get_paginator("list_objects_v2")
    clips: dict[str, list[str]] = {}
    for page in pag.paginate(Bucket=env["R2_BUCKET"], Prefix="track-h/clips/anwm-seeds/C/"):
        for o in page.get("Contents", []):
            if not o["Key"].endswith(".png"):
                continue
            parts = o["Key"].split("/")
            clips.setdefault(parts[4], []).append(o["Key"])
    return [{"source": "h01", "clip_id": k, "stride": 1, "f_px": F_GEN,
             "frames": sorted(v, key=lambda s: int(Path(s).stem))}
            for k, v in sorted(clips.items())]


def download(plan: dict) -> None:
    from src.utils.r2 import client, load_env
    env = load_env(".env")
    c = client(env)
    total = 0
    todo: list[tuple[str, Path]] = []
    for r in plan["sim"]:
        todo.append((r["video"], OUT / "sim/videos" / f"{r['clip_id']}.mp4"))
        todo.append((r["gt"], OUT / "sim/gt" / f"{r['clip_id']}.npz"))
    for r in plan["h04"]:
        todo.append((r["video"], OUT / "h04" / f"{r['clip_id']}.mp4"))
    for r in plan["h01"]:
        for i, key in enumerate(r["frames"]):
            todo.append((key, OUT / "h01" / r["clip_id"] / f"{i:02d}.png"))
    for key, dst in todo:
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = c.get_object(Bucket=env["R2_BUCKET"], Key=key)["Body"].read()
        dst.write_bytes(body)
        total += len(body)
    print(f"downloaded {len(todo)} objects, {total / 1e6:.1f} MB new")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    plan = {"tii": tii_plan(), "sim": sim_plan(), "h04": h04_plan(), "h01": h01_plan()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    n = {k: len(v) for k, v in plan.items()}
    print("plan:", n)
    if args.download:
        download(plan)
    h = hashlib.sha256((OUT / "plan.json").read_bytes()).hexdigest()
    print("plan sha256:", h)


if __name__ == "__main__":
    main()
