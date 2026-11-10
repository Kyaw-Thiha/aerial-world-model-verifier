"""Aggregate the injected-arm evaluation into the Checkpoint 2 tables.

    python -m experiments.anwm_checkpoint2_tables results/anwm_injected/eval_injected_injected_true.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ev = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    idx = {r["clip_id"]: r for r in csv.DictReader(
        open(ROOT / "data/processed/anwm_clips_injected/clips_injected_index.csv"))}
    rows = []
    for cid, e in ev.items():
        r = idx.get(cid, {})
        rows.append({"id": cid, "maneuver": e["maneuver"], "tilt": e["designed_tilt_deg"],
                     "lpips_lvl": e["lpips_vs_level_mean"], "lpips_prior": e["lpips_vs_prior_mean"],
                     "mae_lvl": e["mae_vs_level_mean"], "mae_prior": e["mae_vs_prior_mean"],
                     "cov_min": e["prior_coverage_min"], "cov_mean": e["prior_coverage_mean"],
                     "std": e["frame_std_mean"], "dt": e["temporal_delta"],
                     "repeat": r.get("repeat_noise_mae")})
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["maneuver"], r["tilt"])].append(r)
    print(f"{'cell':22s} {'n':>3s} {'lpips(lvl)':>10s} {'lpips(prior)':>12s} "
          f"{'mae(lvl)':>9s} {'mae(prior)':>10s} {'cov min':>8s} {'cov mean':>8s}")
    for k in sorted(g, key=lambda x: (str(x[0]), -1 if x[1] is None else x[1])):
        a = g[k]
        print(f"{k[0]+' '+('ctl' if k[1] is None else str(int(k[1]))):22s} {len(a):3d} "
              f"{np.mean([x['lpips_lvl'] for x in a]):10.3f} "
              f"{np.mean([x['lpips_prior'] for x in a]):12.3f} "
              f"{np.mean([x['mae_lvl'] for x in a]):9.2f} "
              f"{np.mean([x['mae_prior'] for x in a]):10.2f} "
              f"{min(x['cov_min'] for x in a):8.3f} "
              f"{np.mean([x['cov_mean'] for x in a]):8.3f}")
    if rows:
        print(f"\nall: n={len(rows)} lpips(lvl) {np.mean([r['lpips_lvl'] for r in rows]):.3f} "
              f"lpips(prior) {np.mean([r['lpips_prior'] for r in rows]):.3f}")
        # strata by coverage
        lo = [r for r in rows if r["cov_min"] < 0.2]
        hi = [r for r in rows if r["cov_min"] >= 0.2]
        for name, s in (("cov_min<0.2", lo), ("cov_min>=0.2", hi)):
            if s:
                print(f"  {name}: n={len(s)} lpips(prior) "
                      f"{np.mean([x['lpips_prior'] for x in s]):.3f} "
                      f"lpips(lvl) {np.mean([x['lpips_lvl'] for x in s]):.3f}")
        rep = [r for r in rows if r["repeat"]]
        if rep:
            print(f"  repeat-noise rows: n={len(rep)} mean {np.mean([float(r['repeat']) for r in rep]):.3f}/255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
