"""Task 3, input side: turn Track B's clip set into specs the GPU worker can run.

Track B's layout (``docs/track_b_clips.md`` in that branch)::

    <clips_dir>/clips_index.csv
    <clips_dir>/<regime>/<clip_id>/frames/00.png .. NN.png
    <clips_dir>/<regime>/<clip_id>/manifest.json

Each manifest holds the commanded path (``commanded_path.positions``, 17 rows for 16
frames at 1 s) so the required tilt at any dt is analysis, not generation. This script
writes one npz per clip with the frame names and the implied tilt for the swept dt, and
one ``specs.json`` index; the GPU worker then only decodes frames and runs GeoCalib.

    pixi run -e gpu python experiments/phase3a_anwm_specs.py --clips data/raw/track_b_clips
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.physics.flatness import G0, implied_tilt_from_positions

ROOT = Path(__file__).resolve().parents[1]
DTS = (0.25, 0.667, 1.0)
SG_WINDOW = 11
SG_POLY = 3
K_D = 0.295


def implied_series(points: np.ndarray, dt: float) -> np.ndarray:
    tilt, _ = implied_tilt_from_positions(np.asarray(points, dtype=np.float64), float(dt),
                                          SG_WINDOW, SG_POLY, K_D, G0)
    return tilt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/raw/track_b_clips")
    ap.add_argument("--out", default="data/processed/anwm_specs")
    ap.add_argument("--tag", default="clips_v1")
    args = ap.parse_args()

    clips_dir = ROOT / args.clips
    if not (clips_dir / "clips_index.csv").exists():
        raise SystemExit(f"no clips_index.csv under {clips_dir}: Track B has not delivered")
    out = ROOT / args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # Track B's own index carries a peak required tilt at 1 s; keep it so the two
    # computations can be cross-checked rather than assumed equal
    their_peaks = {}
    idx_path = clips_dir / "clips_index.csv"
    if idx_path.exists():
        import csv
        with idx_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                for k in ("required_tilt_1s_kd_deg", "required_tilt_1s_drag_free_deg"):
                    if row.get(k):
                        their_peaks.setdefault(row["clip_id"], {})[k] = float(row[k])

    index = []
    for man in sorted(clips_dir.glob("*/*/manifest.json")):
        m = json.loads(man.read_text())
        clip_dir = man.parent
        frames = sorted(str(p.name) for p in (clip_dir / "frames").glob("*.png"))
        pos = np.asarray(m["commanded_path"]["positions"], dtype=np.float64)
        if pos.shape[1] == 2:                       # some paths carry (x, y) only
            pos = np.column_stack([pos, np.zeros(len(pos))])
        row = {"clip_id": m.get("id", clip_dir.name), "regime": m.get("regime"),
               "dir": str(clip_dir.relative_to(clips_dir)), "n_frames": len(frames),
               "seed": m.get("seed"), "designed_required_tilt_deg":
                   m.get("designed_required_tilt_deg"), "frames": frames,
               "tilt_implied_deg": {}}
        for dt in DTS:
            row["tilt_implied_deg"][str(dt)] = implied_series(pos, dt).tolist()
        np.savez_compressed(out / f"{row['clip_id']}.npz",
                            frames=np.array(frames), positions=pos,
                            **{f"tilt_{dt}": np.asarray(row["tilt_implied_deg"][str(dt)])
                               for dt in DTS})
        mine = {str(dt): float(np.nanmax(np.asarray(row["tilt_implied_deg"][str(dt)])))
                for dt in DTS}
        theirs = their_peaks.get(row["clip_id"], {})
        cross = {k: {"mine": mine["1.0"], "theirs": v, "delta": mine["1.0"] - v}
                 for k, v in theirs.items()}
        index.append({k: v for k, v in row.items() if k != "tilt_implied_deg"} |
                     {"tilt_peak_deg": mine, "cross_check_1s": cross})
    (out / "specs.json").write_text(json.dumps(
        {"clips_dir": str(clips_dir), "dts": DTS, "sg_window": SG_WINDOW, "k_d": K_D,
         "clips": index}, indent=1))
    print(f"{len(index)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
