"""H03: run the classical (non-learned) roll estimator on the 150-frame sample.

    pixi run -e gpu python experiments/h03_classical.py

Reads ``data/h03_frames/<source>/<clip>/<idx>.png`` and writes
``results/h03/classical.jsonl`` with one row per frame: the chosen reading, both candidate
readings, their supports and the abstention reason.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.perception.horizon import estimate_roll

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/h03_frames"
OUT = ROOT / "results/h03/classical.jsonl"


def main() -> int:
    import cv2

    rows = []
    for p in sorted(FRAMES.glob("*/*/*.png")):
        source, clip = p.parts[-3], p.parts[-2]
        img = cv2.imread(str(p))
        reading, both = estimate_roll(img) if img is not None else (None, {})
        row = {"frame_id": f"{source}/{clip}/{p.stem}", "source": source, "clip_id": clip,
               "frame_index": int(p.stem),
               "roll_deg": None if reading is None or not reading.usable else reading.roll_deg,
               "method": None if reading is None else reading.method,
               "support": None if reading is None else reading.support,
               "n_inliers": None if reading is None else reading.n_inliers,
               "image_ok": img is not None,
               "sky_roll_deg": both.get("sky", {}).get("roll_deg"),
               "sky_support": both.get("sky", {}).get("support"),
               "sky_inliers": both.get("sky", {}).get("n_inliers"),
               "line_roll_deg": both.get("lines", {}).get("roll_deg"),
               "line_support": both.get("lines", {}).get("support"),
               "line_n": both.get("lines", {}).get("n_inliers")}
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["roll_deg"] is not None]
    by = {}
    for r in rows:
        d = by.setdefault(r["source"], [0, 0])
        d[0] += 1
        d[1] += int(r["roll_deg"] is not None)
    print(json.dumps({"rows": len(rows), "usable": len(ok),
                      "by_source": {k: {"n": v[0], "usable": v[1]} for k, v in by.items()},
                      "method_used": {m: sum(1 for r in ok if r["method"] == m)
                                      for m in {r["method"] for r in ok}}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
