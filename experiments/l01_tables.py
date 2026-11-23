"""Render the L01 report's markdown tables from the stored results (no computation).

    PYTHONPATH=. python experiments/l01_tables.py > /tmp/l01_tables.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

L01 = Path("results/l01")


def main() -> int:
    cells = list(csv.DictReader((L01 / "cells.csv").open()))
    fams = list(csv.DictReader((L01 / "families.csv").open()))
    an = json.loads((L01 / "anatomy.json").read_text())
    resc = json.loads((L01 / "reader_rescore.json").read_text())

    print("### The 3 x 4 grid (feasible development clips)\n")
    print("| family | mount | clips | frames | median error, deg | clip range | bias fwd, deg | bias left, deg | >10 deg | gate kept |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in cells:
        bf = float(r["mean_bias_forward"])
        bl = float(r["mean_bias_left"])
        import math
        bfd = math.degrees(math.asin(max(-1, min(1, bf))))
        bld = math.degrees(math.asin(max(-1, min(1, bl))))
        print(f"| {r['family']} | {float(r['uptilt_deg']):.0f} | {r['clips']} | {r['frames']} | "
              f"{float(r['median_of_clip_medians']):.2f} | {float(r['min_clip_median']):.2f}-"
              f"{float(r['max_clip_median']):.2f} | {bfd:+.2f} | {bld:+.2f} | "
              f"{100*float(r['frac_gt10_frameweighted']):.1f} % | "
              f"{r['gate_keep']}/{r['gate_n']} |")

    print("\n### By family\n")
    print("| family | clips | frames | median error | bias fwd | bias left | >10 deg | >20 deg | gate kept |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in fams:
        import math
        bf = math.degrees(math.asin(max(-1, min(1, float(r["mean_bias_forward"])))))
        bl = math.degrees(math.asin(max(-1, min(1, float(r["mean_bias_left"])))))
        print(f"| {r['family']} | {r['clips']} | {r['frames']} | "
              f"{float(r['median_of_clip_medians']):.2f} | {bf:+.2f} | {bl:+.2f} | "
              f"{100*float(r['frac_gt10_frameweighted']):.1f} % | "
              f"{100*float(r['frac_gt20_frameweighted']):.1f} % | {r['gate_keep']}/{r['gate_n']} |")
    print("\nFamily spread per mount (max - min of the three family medians):")
    for s in resc["spread"]:
        print(f"- {float(s['uptilt_deg']):.0f} deg mount: {s['max_minus_min']:.2f} deg "
              f"(values {', '.join(f'{v:.2f}' for v in s['values'])})")

    print("\n### The frozen S3 gate\n")
    print("| family | n | kept | keep rate | kept median | dropped median | kept >10 deg |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for g in resc["gate"]:
        km = g["kept_median_of_clip_medians"]
        dm = g["dropped_median_of_clip_medians"]
        print(f"| {g['family']} | {g['n']} | {g['kept']} | {g['keep_rate']:.3f} | "
              f"{km if km is None else f'{km:.2f}'} | {dm if dm is None else f'{dm:.2f}'} | "
              f"{100*g['kept_frac_gt10']:.1f} % |")

    print("\n### Anatomy (J01's constant / slow / fast), shares of the total second moment\n")
    print("| family | pooled bias | pooled slow | pooled fast | per-clip bias (median) | per-clip slow | per-clip fast |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in an:
        p, c = row["pooled"], row["per_clip"]
        print(f"| {row['family']} | {100*p['bias']:.1f} % | {100*p['slow']:.1f} % | "
              f"{100*p['fast']:.1f} % | {100*c['bias_median']:.1f} % | "
              f"{100*c['slow_median']:.1f} % | {100*c['fast_median']:.1f} % |")

    rep = resc["reproduction"]
    print(f"\nReproduction: max |delta| {rep['max_abs_delta']:.3f} deg, p90 "
          f"{rep['p90_abs_delta']:.3f}, pooled median delta {rep['pooled_median_delta']:.4f}, "
          f"{rep['n_over_0.05_original']} above the original 0.05, "
          f"{rep['n_over_0.25_amended']} above the amended 0.25.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
