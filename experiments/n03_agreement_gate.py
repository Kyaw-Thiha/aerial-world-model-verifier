"""Track N, Part 3: the reader-vs-propagation agreement gate and its coverage.

Pre-registration ``configs/n03_gate_prereg.yaml`` (before any gate number). Reads N02b's per-clip
table; no GPU.

    PYTHONPATH=. python experiments/n03_agreement_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n03"
N02B = ROOT / "results/n02b"
NO_VIOL = ("real", "null_edit")
VIOL = ("tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll")
TAUS = np.round(np.arange(0.0, 20.01, 0.5), 2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(N02B / "per_clip.csv")
    d = d[d["valid_extraction"].astype(bool) & (d["peak_tilt_deg"] >= 15.0)].copy()
    seqs = sorted(d["seq"].unique())
    # per-clip thresholds: leave-one-sequence-out 95th percentile of no-violation scores
    thr = {}
    for key in ("F0_reader", "F0_propagated", "R_reader", "R_propagated"):
        thr[key] = {}
        for s in seqs:
            other = d[(d.seq != s) & d.condition.isin(NO_VIOL)][key].dropna()
            thr[key][s] = float(np.quantile(other, 0.95)) if len(other) else np.nan

    def flagged(row, key):
        t = thr[key].get(row["seq"], np.nan)
        v = row[key]
        return bool(np.isfinite(v) and np.isfinite(t) and v > t)

    d["flag_F0_reader"] = d.apply(lambda r: flagged(r, "F0_reader"), axis=1)
    d["flag_F0_propagated"] = d.apply(lambda r: flagged(r, "F0_propagated"), axis=1)
    d["flag_R_reader"] = d.apply(lambda r: flagged(r, "R_reader"), axis=1)
    d["flag_R_propagated"] = d.apply(lambda r: flagged(r, "R_propagated"), axis=1)

    summary = {"n_dynamic_valid": len(d), "clusters": len(seqs),
               "thresholds_loso": thr, "gates": {}}
    for target in (0.10, 0.20):
        # risk-coverage curve with the gate; report coverage and risk at the pre-registered
        # conformal selection: smallest tau whose pooled no-violation rate <= target
        best = None
        for tau in TAUS:
            kept = d[d.disagreement_deg <= tau]
            nv = kept[kept.condition.isin(NO_VIOL)]
            if not len(nv):
                continue
            rate = nv["flag_F0_reader"].mean()
            if rate <= target:
                best = tau
                break
        g = {"target": target, "tau": None}
        if best is not None:
            kept = d[d.disagreement_deg <= best]
            g.update({
                "tau": float(best),
                "coverage": f"{len(kept)}/{len(d)}",
                "coverage_frac": float(len(kept) / len(d)),
                "no_violation_rate_F0_reader": f"{int(kept[kept.condition.isin(NO_VIOL)]['flag_F0_reader'].sum())}/{len(kept[kept.condition.isin(NO_VIOL)])}",
                "detection_F0_reader": {c: f"{int(kept[kept.condition == c]['flag_F0_reader'].sum())}/{len(kept[kept.condition == c])}" for c in VIOL},
                "detection_F0_propagated": {c: f"{int(kept[kept.condition == c]['flag_F0_propagated'].sum())}/{len(kept[kept.condition == c])}" for c in VIOL},
                "per_sequence_kept": {s: f"{int((kept.seq == s).sum())}/{int((d.seq == s).sum())}" for s in seqs},
            })
        summary["gates"][str(target)] = g

    # ungated reference and the full curve
    curve = []
    for tau in TAUS:
        kept = d[d.disagreement_deg <= tau]
        nv = kept[kept.condition.isin(NO_VIOL)]
        curve.append({
            "tau": float(tau), "coverage_frac": float(len(kept) / len(d)),
            "no_violation_flags_F0_reader": int(nv["flag_F0_reader"].sum()),
            "no_violation_n": int(len(nv)),
            "detection_F0_reader_removal": f"{int(kept[kept.condition=='tilt_removal']['flag_F0_reader'].sum())}/{int((kept.condition=='tilt_removal').sum())}",
            "detection_F0_reader_mirror": f"{int(kept[kept.condition=='mirror_roll']['flag_F0_reader'].sum())}/{int((kept.condition=='mirror_roll').sum())}",
        })
    pd.DataFrame(curve).to_csv(OUT / "curve.csv", index=False)
    summary["ungated"] = {
        "n": len(d),
        "no_violation_flags_F0_reader": f"{int(d[d.condition.isin(NO_VIOL)]['flag_F0_reader'].sum())}/{int(d.condition.isin(NO_VIOL).sum())}",
        "detection_F0_reader": {c: f"{int(d[d.condition==c]['flag_F0_reader'].sum())}/{int((d.condition==c).sum())}" for c in VIOL},
        "coverage_frac": 1.0,
    }
    summary["note"] = ("descriptive; 3 sequences; a 0.05 target is not certifiable with 6 "
                       "no-violation clips, so 0.10 and 0.20 are the reported targets")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["gates"], indent=1))
    print("ungated", json.dumps(summary["ungated"]))
    return 0


def _thr(d, key, subset):
    # helper for the curve rows only; per-clip flags use the LOSO thresholds above
    return np.nan


if __name__ == "__main__":
    raise SystemExit(main())
