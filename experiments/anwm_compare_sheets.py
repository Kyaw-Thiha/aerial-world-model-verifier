"""Level-versus-injected contact sheets for the Checkpoint 2 designs.

    python -m experiments.anwm_compare_sheets C_turn_20_fixed_0 C_accel_brake_20_course_0 C_slalom_10_fixed_0

One sheet per clip: level frames (top row), injected frames (bottom row), 16 steps each; plus a
third row with the injected prior, so a reader can see what the model was conditioned on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "data/processed/anwm_clips/C"
REGEN = ROOT / "data/processed/anwm_clips_regen/C"
INJ = ROOT / "data/processed/anwm_clips_injected/injected_true/C"
OUT = ROOT / "results/anwm_injected/sheets"


def frames(d: Path, sub: str = "frames") -> list[np.ndarray]:
    return [np.asarray(Image.open(p)) for p in sorted((d / sub).glob("*.png"))]


def main() -> int:
    ids = sys.argv[1:] or ["C_turn_20_fixed_0", "C_accel_brake_20_course_0", "C_slalom_10_fixed_0"]
    OUT.mkdir(parents=True, exist_ok=True)
    for cid in ids:
        lvl_dir = REGEN / cid if (REGEN / cid).exists() else LEVEL / cid
        lvl, inj = frames(lvl_dir), frames(INJ / cid)
        pri = frames(INJ / cid, "priors")
        n = min(len(lvl), len(inj), len(pri))
        h, w = inj[0].shape[:2]
        sheet = Image.new("RGB", (w * n, h * 3))
        for i in range(n):
            sheet.paste(Image.fromarray(lvl[i][..., :3]), (i * w, 0))
            sheet.paste(Image.fromarray(inj[i][..., :3]), (i * w, h))
            sheet.paste(Image.fromarray(pri[i][..., :3]), (i * w, 2 * h))
        p = OUT / f"{cid}_level_vs_injected.png"
        sheet.save(p)
        d = float(np.abs(np.asarray(lvl[0], float) - np.asarray(inj[0], float)).mean())
        print(f"{cid:34s} {p.relative_to(ROOT)}  frame MAE step0 {d:.1f}/255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
