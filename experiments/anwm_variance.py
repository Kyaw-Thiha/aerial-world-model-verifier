"""Sampling variability across seeds for the clips that were generated with 3 seeds."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/anwm_clips"


def load(d: Path) -> np.ndarray:
    fs = sorted((d / "frames").glob("*.png"))
    return np.stack([np.asarray(Image.open(f), float) / 255.0 for f in fs])


def main() -> int:
    groups: dict[str, list[Path]] = {}
    for d in sorted(CLIPS.glob("*/*")):
        m = re.match(r"(.*)_s\d+$", d.name)
        key = m.group(1) if m else d.name
        groups.setdefault(key, []).append(d)
    out = {}
    for key, dirs in groups.items():
        if len(dirs) < 2:
            continue
        arrs = [load(d) for d in dirs]
        n = min(a.shape[0] for a in arrs)
        maes = [float(np.abs(a[:n] - b[:n]).mean())
                for i, a in enumerate(arrs) for b in arrs[i + 1:]]
        per_step = [float(np.abs(a[i] - b[i]).mean()) for a in arrs for b in arrs
                    if a is not b for i in range(n)]
        out[key] = {"n_seeds": len(arrs), "pairwise_mae": maes,
                    "mean_pairwise_mae": float(np.mean(maes)),
                    "per_step_mae_mean": float(np.mean(per_step)),
                    "per_step_mae_max": float(np.max(per_step)),
                    "within_clip_temporal_delta": float(np.mean(
                        [np.abs(a[1:] - a[:-1]).mean() for a in arrs]))}
    (CLIPS / "variance.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k}: seeds {v['n_seeds']} pairwise MAE {v['mean_pairwise_mae']:.4f} "
              f"temporal {v['within_clip_temporal_delta']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
