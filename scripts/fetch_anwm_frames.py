"""Fetch a small sample of released ANWM frames (level camera) by HTTP Range.

Used for the GeoCalib noise-floor test (handoff 04, Task 1): the released frames are
known to be level (roll = pitch = 0), so the estimated tilt on them is the false-alarm
floor of the estimator. Layout of a shard (verified): for each trajectory directory the
tar carries `traj_data.pkl` first and then the numbered JPEGs, so a trajectory's frames
start right after the pickle's data blocks. `data/raw/anwm/shard_index.json` already has
[shard, pickle_data_offset, pickle_size] per trajectory.

Cost: one 512 B header read per member walked plus the bytes of the selected JPEGs.

    pixi run -e gpu python scripts/fetch_anwm_frames.py --n-traj 16 --per-traj 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.fetch_anwm import RangeFile, shard_url

OUT = Path("data/raw/anwm_frames")
INDEX = Path("data/raw/anwm/shard_index.json")
PROCESSED = Path("data/raw/anwm")


def fetch_trajectory(traj: str, entry: list, per_traj: int, stride: int) -> list[str]:
    shard, off, size = entry
    rf = RangeFile(shard_url(shard))
    pos = off + size + ((-size) % 512)  # first member after the pickle
    out_dir = OUT / traj
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    seen = 0
    while len(saved) < per_traj and pos < rf.size:
        hdr = rf.read_range(pos, 512)
        if len(hdr) < 512 or hdr[257:262] != b"ustar":
            break
        name = hdr[:100].split(b"\x00")[0].decode()
        if not name.startswith(traj):  # next trajectory directory reached
            break
        nbytes = int(hdr[124:136].split(b"\x00")[0].strip() or b"0", 8)
        data_lo = pos + 512
        if name.endswith(".jpg"):
            if seen % stride == 0:
                (out_dir / Path(name).name).write_bytes(rf.read_range(data_lo, nbytes))
                saved.append(Path(name).name)
            seen += 1
        pos = data_lo + nbytes + ((-nbytes) % 512)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=16)
    ap.add_argument("--per-traj", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3, help="keep every k-th member walked")
    args = ap.parse_args()
    index = json.loads(INDEX.read_text())
    test_ids = sorted(p.name.replace(".npz", "").replace("_processed", "") + "_processed"
                      for p in PROCESSED.glob("*_processed.npz"))
    avail = [t for t in test_ids if t in index]
    step = max(1, len(avail) // args.n_traj)
    chosen = avail[::step][: args.n_traj]
    manifest = {}
    for t in chosen:
        names = fetch_trajectory(t, index[t], args.per_traj, args.stride)
        manifest[t] = names
        print(f"{t}: {len(names)} frames", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
