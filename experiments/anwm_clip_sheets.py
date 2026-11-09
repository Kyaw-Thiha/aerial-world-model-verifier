"""Contact sheet (4x4) per generated clip, for the qualitative visual-integrity note."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/anwm_clips"


def main() -> int:
    n = 0
    for d in sorted(CLIPS.glob("*/*")):
        frames = sorted((d / "frames").glob("*.png"))
        if not frames:
            continue
        ims = [Image.open(f) for f in frames]
        w, h = ims[0].size
        cols = 8
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, h * rows), "black")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(d / "strip.png")
        n += 1
    print(f"{n} contact sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
