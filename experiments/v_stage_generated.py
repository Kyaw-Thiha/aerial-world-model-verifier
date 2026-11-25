"""Track V, Part 3 input staging: generated clips whose conditioning path has vertical motion.

ANWM (H01, 16 clips: 4 designed climb_turn paths x seeds 1-4) and SolarWM (H04, 18 clips:
2 climb_turn paths x 3 arms x 3 draws). Frames and the conditioning path come from the Modal
volumes the generating tracks wrote; nothing is regenerated.

The generated clips have no metric truth, so only the scale-free statistic ``rho = dz/L`` is
used (``configs/v10_vertical_prereg.yaml`` Part 3, amendment 1). Intrinsics are unknown: a 90 deg
horizontal field of view is assumed (the convention this project already used for generated
video in Tracks N02/N04), and the focal choice is recorded per unit.

Usage
-----
PYTHONPATH=. python experiments/v_stage_generated.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v"
FRAMES = OUT / "frames"
UNITS = OUT / "units.json"
HFOV_ASSUMED_DEG = 90.0


def _hfov_k(w: int, h: int) -> dict:
    fx = (w / 2.0) / np.tan(np.radians(HFOV_ASSUMED_DEG) / 2.0)
    return {"fx": float(fx), "fy": float(fx), "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0,
            "w": int(w), "h": int(h), "assumed_hfov_deg": HFOV_ASSUMED_DEG}


def stage_anwm() -> list[dict]:
    units = []
    for d in sorted((OUT / "gen" / "anwm").iterdir()):
        man = json.loads((d / "manifest.json").read_text())
        pos = np.asarray(man["commanded_path"]["positions"], float)
        frames = sorted((d / "frames").glob("*.png"))
        if len(frames) != man["frames"]:
            print(f"WARN {d.name}: {len(frames)} frames vs manifest {man['frames']}")
        dst = FRAMES / "anwm" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(frames):
            out = dst / f"{k:04d}.png"
            if out.exists():
                out.unlink()
            out.write_bytes(p.read_bytes())
        from PIL import Image
        w, h = Image.open(frames[0]).size
        n = len(frames)
        t = np.arange(n) / 3.0            # only ordering matters for the whole-clip window
        units.append({"unit": d.name, "kind": "anwm", "frames": str(dst.relative_to(FRAMES)),
                      "n_frames": n, "K": _hfov_k(w, h), "fps": 3.0,
                      "truth": {"t": t.tolist(), "pos": pos.tolist(),
                                "z_axis": "project z-up; designed path climbs",
                                "maneuver": man["meta"]["maneuver"],
                                "base_path": man["meta"]["base_path"],
                                "arm": man.get("arm")},
                      "source": {"volume": "awmv-h-anwm-results",
                                 "path": f"clips/C/{d.name}"}})
    return units


def stage_solarwm() -> list[dict]:
    units = []
    for d in sorted((OUT / "gen" / "solarwm").iterdir()):
        man = json.loads((d / "manifest.json").read_text())
        pos = np.asarray(man["commanded_path"]["positions"], float)
        src_frames = sorted((d / "frames").glob("*.png"))
        dst = FRAMES / "solarwm" / d.name
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        kept = src_frames[::2]             # 81 frames -> 41 at ~7.5 Hz
        for k, p in enumerate(kept):
            out = dst / f"{k:04d}.png"
            out.write_bytes(p.read_bytes())
        from PIL import Image
        w, h = Image.open(kept[0]).size
        n = len(kept)
        t = np.arange(n) / 7.5
        units.append({"unit": d.name, "kind": "solarwm", "frames": str(dst.relative_to(FRAMES)),
                      "n_frames": n, "K": _hfov_k(w, h), "fps": 7.5,
                      "truth": {"t": t.tolist(), "pos": pos.tolist(),
                                "z_axis": "project z-up; designed path climbs",
                                "maneuver": man.get("meta", {}).get("maneuver", "climb_turn"),
                                "arm": man.get("arm")},
                      "source": {"volume": "awmv-h3-results",
                                 "path": f"solarwm/main/{d.name}"}})
    return units


def main() -> int:
    new = stage_anwm() + stage_solarwm()
    old = json.loads(UNITS.read_text())["units"] if UNITS.exists() else []
    seen = {u["unit"] for u in new}
    merged = [u for u in old if u["unit"] not in seen] + new
    UNITS.write_text(json.dumps({"generated": "experiments/v_stage_frames.py + "
                                             "experiments/v_stage_generated.py",
                                 "units": merged}, indent=1))
    print(f"{len(new)} generated units added ({len(merged)} total)")
    for u in new:
        p = np.asarray(u["truth"]["pos"], float)
        dz = float(p[-1, 2] - p[0, 2])
        L = float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))
        print(f"  {u['unit']}: {u['n_frames']} frames {u['K']['w']}x{u['K']['h']} "
              f"dz={dz:.2f} m L={L:.1f} m rho_cmd={dz / max(L, 1e-9):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
