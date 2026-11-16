"""Descriptive analysis of the F01 stage-0 rows: separation, thresholds, excitation.

Threshold-free summaries (rank AUC of real vs each edit) are reported beside the
pre-registered rates. They do not change any decision rule; the gate uses the LOSO flags.
Also computes how mirror detection depends on the clip's roll excitation, which is the
mechanical explanation of the mirror rate.

    python experiments/f01_analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PRIMARY_SEQS = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_9",
                "indoor_forward_10")
VARIANTS = ("V1_preintegration", "V2_shooting", "V3_band_coherence",
            "V4_velocity_drag", "V5_physics_scale")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC that a value from ``pos`` exceeds one from ``neg`` (ties count half)."""
    pos = np.asarray([v for v in pos if np.isfinite(v)], dtype=float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            mean_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = mean_rank
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main() -> int:
    rows = list(csv.DictReader(open("results/f01_stage0/rows.csv")))

    def val(r, key="statistic"):
        if not r:
            return float("nan")
        try:
            v = float(r[key])
        except (TypeError, ValueError):
            return float("nan")
        return v

    out = {"note": "descriptive; the gate uses the pre-registered LOSO flags", "auc": {},
           "mirror_vs_roll_excitation": {}}
    for arm, path in (("estimated_attitude", "pixel"), ("estimated_attitude", "oracle"),
                      ("oracle_attitude", "pixel")):
        for variant in VARIANTS:
            sub = [r for r in rows if r["variant"] == variant and r["arm"] == arm
                   and r["path"] == path and r["testable"] == "True"
                   and (r["seq"], r["idx"]) in {(x["seq"], x["idx"]) for x in rows
                                                if x["cond"] == "real"
                                                and x["seq"] in PRIMARY_SEQS
                                                and x["full_path"] == "True"}]
            real = [val(r) for r in sub if r["cond"] == "real"]
            entry = {}
            for cond in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
                         "null_edit", "yaw_control"):
                edit = [val(r) for r in sub if r["cond"] == cond]
                entry[cond] = auc(edit, real)
            out["auc"][f"{variant}|{arm}|{path}"] = entry

    # mirror detection against the clip's roll excitation (depicted lean roll spread)
    key = "V2_shooting|estimated_attitude|pixel"
    sub = [r for r in rows if r["variant"] == "V2_shooting"
           and r["arm"] == "estimated_attitude" and r["path"] == "pixel"
           and r["seq"] in PRIMARY_SEQS and r["full_path"] == "True"]
    real = {(r["seq"], r["idx"]): r for r in sub if r["cond"] == "real"}
    mirror = {(r["seq"], r["idx"]): r for r in sub if r["cond"] == "mirror_roll"}
    removal = {(r["seq"], r["idx"]): r for r in sub if r["cond"] == "tilt_removal"}
    table = []
    for key, r in sorted(real.items()):
        clip = r["clip"]
        m = mirror.get(key, {})
        d = removal.get(key, {})
        table.append({
            "clip": clip,
            "roll_spread": float(r["roll_spread"] or "nan"),
            "pitch_spread": float(r["pitch_spread"] or "nan"),
            "stat_real": val(r),
            "stat_removal": val(d),
            "stat_mirror": val(m),
            "flag_removal": d.get("flag"),
            "flag_mirror": m.get("flag"),
        })
    out["mirror_vs_roll_excitation"] = {"variant": key, "clips": table}
    Path("results/f01_stage0/analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["auc"], indent=1))
    print("\nmirror / roll excitation (V2, estimated attitude, pixel path):")
    for c in table:
        print(f"  {c['clip']:32s} roll_sd={c['roll_spread']:.4f} "
              f"pitch_sd={c['pitch_spread']:.4f} real={c['stat_real']:.3f} "
              f"removal={c['stat_removal']:.3f}({c['flag_removal']}) "
              f"mirror={c['stat_mirror']:.3f}({c['flag_mirror']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
