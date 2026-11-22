"""Pull only clip.mp4 from a SolarWM/LingBot output tree and add the sample meta.json.

    pixi run python scripts/h02_pull_mp4.py --root solarwm/grid --dest data/h02_solarwm_grid_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h02_solarwm_inputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--with-meta", action="store_true")
    args = ap.parse_args()
    vol = modal.Volume.from_name("awmv-h2-results")
    dest = ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in vol.listdir("/" + args.root):
        if e.type.name != "DIRECTORY":
            continue
        sid = e.path.split("/")[-1]
        try:
            data = b"".join(vol.read_file(f"{e.path}/clip.mp4"))
        except Exception:
            continue
        d = dest / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "clip.mp4").write_bytes(data)
        if args.with_meta and (INPUTS / sid / "meta.json").exists():
            (d / "meta.json").write_text((INPUTS / sid / "meta.json").read_text())
        n += 1
    print(f"{n} clips -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
