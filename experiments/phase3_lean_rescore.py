"""Phase 3a (handoff 04, Task 0.2): re-score lean direction with windowed azimuth alignment.

The Phase 1b lean number (median 83.6 deg) used **one constant azimuth per clip** to
absorb the VO frame's arbitrary heading. The verifier never compares world-frame
azimuths over a clip: it compares the depicted axis with the implied axis inside one
frame. Under that reading, a slowly drifting yaw is not an error, so this run re-scores
the same per-frame axes with one fitted azimuth per sliding window and reports both
versions side by side. Nothing is refitted: it consumes the saved Phase 1b arrays.

Input : results/phase1b_fusion/<run>/frames_<flight>.npz (z_est, z_true, tilt_*)
Output: results/phase3a_verdict/<run_id>_lean/{leans.json, lean_windows.png, config.yaml}
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.fusion import lean_error_deg, lean_error_windowed_deg

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_S = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)  # handoff asked for 3-5 s; the sweep shows where the error actually lives
STRIDE = 10          # the GeoCalib/DPVO runs this re-score consumes
FPS = 120.4          # measured camera rate (Phase 1); only sets the window duration


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def rescore_one(z: np.lib.npyio.NpzFile) -> dict:
    est, tru = z["z_est"], z["z_true"]
    t = np.arange(len(est)) * STRIDE / FPS
    out = {"n_frames": int(len(est)),
           "one_per_clip": {k: v for k, v in lean_error_deg(est, tru, 10.0).items()},
           "windowed": {}}
    for w in WINDOWS_S:
        r = lean_error_windowed_deg(est, tru, t, win_s=w, min_tilt_deg=10.0)
        out["windowed"][f"{w:g}"] = {k: v for k, v in r.items() if k != "residual_deg"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-run", default="results/phase1b_fusion/20261108T133135Z_c738aa8")
    ap.add_argument("--out", default="results/phase3a_verdict")
    ap.add_argument("--tag", default="lean")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / f"{run_id}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    per_flight: dict = {}
    for p in sorted((ROOT / args.fusion_run).glob("frames_*.npz")):
        flight = p.name.replace("frames_", "").replace(".npz", "")
        with np.load(p) as z:
            per_flight[flight] = rescore_one(z)
        r = per_flight[flight]
        print(f"{flight}: one-per-clip {r['one_per_clip'].get('median_abs_deg', float('nan')):.1f} "
              f"-> windowed "
              + ", ".join(f"{w}s {v.get('median_abs_deg', float('nan')):.1f}"
                          for w, v in r["windowed"].items()), flush=True)

    def pooled(key: str, sub: str | None) -> float:
        vals = [v[key].get("median_abs_deg") if sub is None
                else v["windowed"][sub].get("median_abs_deg") for v in per_flight.values()]
        vals = [x for x in vals if x is not None]
        return float(np.median(vals)) if vals else float("nan")

    summary = {
        "source_run": args.fusion_run,
        "note": "medians are over flights of the per-flight median absolute lean error; "
                "one_per_clip is the Phase 1b number, windowed is the re-score",
        "one_per_clip_median_deg": pooled("one_per_clip", None),
        "windowed_median_deg": {f"{w:g}": pooled("windowed", f"{w:g}") for w in WINDOWS_S},
        "gate1_lean_target_deg": 15.0,
    }
    summary["windowed_pass"] = {k: bool(v < 15.0) for k, v in summary["windowed_median_deg"].items()}
    (out / "leans.json").write_text(json.dumps({"summary": summary, "per_flight": per_flight},
                                               indent=1))
    (out / "git.txt").write_text(git_hash() + "\n")

    flights = sorted(per_flight)
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.45 * len(flights)))
    y = np.arange(len(flights))
    ax.barh(y - 0.27, [per_flight[f]["one_per_clip"]["median_abs_deg"] for f in flights],
            height=0.25, label="one azimuth per clip", color="#b23b3b")
    for i, w in enumerate(WINDOWS_S):
        ax.barh(y - 0.27 + 0.25 * (i + 1),
                [per_flight[f]["windowed"][f"{w:g}"]["median_abs_deg"] for f in flights],
                height=0.25, label=f"window {w:g} s", alpha=0.85)
    ax.axvline(15.0, color="k", ls="--", lw=1, label="Gate 1 target 15 deg")
    ax.set_yticks(y)
    ax.set_yticklabels(flights, fontsize=8)
    ax.set_xlabel("median absolute lean-direction error (deg)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "lean_windows.png", dpi=140)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
