"""H03: draw the three roll readings as horizon lines on one frame, for the eye check.

    PYTHONPATH=. pixi run -e gpu python experiments/h03_overlay.py --source lingbot_bf16

Writes ``results/h03/overlays/<frame>.png``: the reader's horizon in green, Perspective
Fields' in blue, the classical reading in red, each a line through the image centre at the
estimated horizon slope (y down), plus the numbers burnt into the corner.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lingbot_bf16")
    ap.add_argument("--frames", default="last,first")
    args = ap.parse_args()
    import cv2

    rows = list(csv.DictReader((ROOT / "results/h03/frames_scored.csv").open()))
    sub = [r for r in rows if r["source"] == args.source]
    picks = []
    for clip in sorted({r["clip_id"] for r in sub}):
        cr = sorted([r for r in sub if r["clip_id"] == clip],
                    key=lambda r: int(r["frame_index"]))
        if "last" in args.frames:
            picks.append(cr[-1])
        if "first" in args.frames:
            picks.append(cr[0])
    out = ROOT / "results/h03/overlays"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for r in picks:
        img = cv2.imread(str(ROOT / "data/h03_frames" / r["source"] / r["clip_id"]
                              / f"{int(r['frame_index']):03d}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        for key, colour in (("reader_roll", (0, 255, 0)), ("pf_roll", (255, 200, 0)),
                            ("classical_roll", (0, 0, 255))):
            v = r.get(key)
            if v in ("", None):
                continue
            ang = np.deg2rad(float(v))
            d = np.array([np.cos(ang), np.sin(ang)])
            p0 = (int(cx - d[0] * w), int(cy - d[1] * w))
            p1 = (int(cx + d[0] * w), int(cy + d[1] * w))
            cv2.line(img, p0, p1, colour, 2)
        txt = (f"reader {float(r['reader_roll']):+.1f}  "
               f"pf {float(r['pf_roll']):+.1f}  "
               f"class {r['classical_roll'] or 'nan'}  "
               f"({r['classical_method']})")
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        name = f"{r['clip_id']}_{int(r['frame_index']):03d}.png"
        cv2.imwrite(str(out / name), img)
        written.append(name)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
