"""Stage 0, item 5: is generation deterministic for a fixed (path, seed, checkpoint, steps)?

Compares regenerated clips (Modal volume, /r/regen_stage0) with the delivered level clips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DELIVERED = ROOT / "data/processed/anwm_clips/C"
REGEN = ROOT / "data/processed/anwm_clips_regen/C"


def frames(d: Path) -> list[np.ndarray]:
    return [np.asarray(Image.open(p), float) for p in sorted((d / "frames").glob("*.png"))]


def main() -> int:
    rows = {}
    for d in sorted(REGEN.glob("*")):
        a = frames(DELIVERED / d.name)
        b = frames(d)
        if not a or not b:
            print("missing frames for", d.name)
            continue
        n = min(len(a), len(b))
        mad = [float(np.abs(a[i] - b[i]).mean()) for i in range(n)]
        maxd = [float(np.abs(a[i] - b[i]).max()) for i in range(n)]
        frac_gt1 = [float((np.abs(a[i] - b[i]) > 1.0).mean()) for i in range(n)]
        rows[d.name] = {
            "n_frames": n,
            "mean_abs_diff": float(np.mean(mad)),
            "max_abs_diff": float(np.max(maxd)),
            "max_frame_mae": float(np.max(mad)),
            "frac_pixels_diff_gt_1_of_255": float(np.mean(frac_gt1)),
            "per_step_mae": mad,
            "identical": bool(np.all(np.array(maxd) == 0.0)),
        }
    out = ROOT / "results/anwm_injected/stage0_determinism.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    for k, v in rows.items():
        print(f"{k:34s} mae {v['mean_abs_diff']:.4f} max {v['max_abs_diff']:.1f} "
              f"frac>1/255 {v['frac_pixels_diff_gt_1_of_255']:.4f} identical {v['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
