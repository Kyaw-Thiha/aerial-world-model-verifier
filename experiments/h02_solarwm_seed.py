"""H02: SolarWM seed variance (10 paths x 3 seeds, flatness arm).

    pixi run python experiments/h02_solarwm_seed.py --depicted results/h02_solarwm_depicted/depicted.jsonl \
        --out results/h02_solarwm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.a2_anwm_score import depicted_tilt_deg, frame_arrays
from experiments.h02_verdict import required_profile
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world
from src.perception.frames import camera_to_body_pitched

ROOT = Path(__file__).resolve().parents[1]
THRESH = 6.46


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(ln) for ln in (ROOT / args.depicted).read_text().splitlines() if ln.strip()]
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    specs = {s["id"]: s for s in json.loads(
        (ROOT / "results/anwm_clips/specs.json").read_text())}
    man = json.loads((ROOT / "data/h02_solarwm_inputs/manifest.json").read_text())
    per = []
    for sid in man["samples"]:
        if not sid.startswith("seed_"):
            continue
        mp = ROOT / "data/h02_solarwm_inputs" / sid / "meta.json"
        meta = json.loads(mp.read_text())
        rs = sorted(by_clip.get(sid, []), key=lambda r: int(Path(r["frame"]).stem))
        if not rs:
            continue
        dep = depicted_tilt_deg(frame_arrays(rs)["up"])
        req = required_profile(specs[meta["source_clip"]], len(dep))
        n = min(len(dep), len(req))
        resid = float(np.nanmedian(np.abs(req[:n] - dep[:n])))
        per.append({"clip_id": sid, "source_clip": meta["source_clip"],
                    "seed": int(meta["seed"]), "depicted_median_deg":
                    float(np.nanmedian(dep[:n])), "residual_median_deg": resid,
                    "feasible": bool(resid <= THRESH)})
    df = pd.DataFrame(per).sort_values(["source_clip", "seed"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seed_per_clip.csv", index=False)
    spread = []
    for cid, g in df.groupby("source_clip"):
        g = g.sort_values("seed")
        r = g.residual_median_deg.to_numpy(float)
        rN = np.minimum.accumulate(r)
        spread.append({"source_clip": cid, "n_seeds": len(g),
                       "depicted_median_deg": float(g.depicted_median_deg.mean()),
                       "depicted_sd_deg": float(np.std(g.depicted_median_deg, ddof=1)),
                       "depicted_range_deg": float(g.depicted_median_deg.max()
                                                   - g.depicted_median_deg.min()),
                       "residual_median_deg": float(np.mean(r)),
                       "residual_range_deg": float(r.max() - r.min()),
                       "r1_deg": float(rN[0]), "r3_deg": float(rN[-1]),
                       "feasible_any": bool(g.feasible.any())})
    sp = pd.DataFrame(spread)
    sp.to_csv(out / "seed_per_path.csv", index=False)
    # exploratory best-of-N curve (N = 1, 2, 3); the pre-registered test is N = 3
    paths = []
    for cid, g in df.groupby("source_clip"):
        r = g.sort_values("seed").residual_median_deg.to_numpy(float)
        paths.append(r)
    r1 = np.array([r[0] for r in paths])
    curve = []
    for N in (1, 2, 3):
        rn = np.array([r[:N].min() for r in paths])
        curve.append({"N": N,
                      "reduction_median_deg": float(np.median(r1 - rn)),
                      "reduction_mean_deg": float(np.mean(r1 - rn)),
                      "n_flipped": int(((r1 > THRESH) & (rn <= THRESH)).sum())})
    summary = {"n_paths": len(sp), "n_clips": len(df), "best_of_n": curve,
               "depicted_sd_median_deg": float(sp.depicted_sd_deg.median()),
               "depicted_range_median_deg": float(sp.depicted_range_deg.median()),
               "oracle_reduction_median_deg": float((sp.r1_deg - sp.r3_deg).median()),
               "oracle_reduction_mean_deg": float((sp.r1_deg - sp.r3_deg).mean()),
               "n_flipped": int(((sp.r1_deg > THRESH) & (sp.r3_deg <= THRESH)).sum()),
               "n_paths": len(sp)}
    (out / "seed_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(sp.to_string(index=False)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
