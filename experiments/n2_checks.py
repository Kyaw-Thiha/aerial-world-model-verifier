"""Track N2, report-number checks: every headline number in N2A/N2B/N2C against the stored results.

Run after the three experiment scripts:

    PYTHONPATH=. pixi run python experiments/n2_checks.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N07 = ROOT / "results/n2/n07"
N08 = ROOT / "results/n2/n08"
N09 = ROOT / "results/n2/n09"
FAILS: list[str] = []
NCHECKS = 0


def check(name, got, want):
    global NCHECKS
    NCHECKS += 1
    ok = got == want
    print(("ok   " if ok else "FAIL ") + f"{name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(name)


def main() -> int:
    s7 = json.load(open(N07 / "summary.json"))
    c7 = list(csv.DictReader(open(N07 / "census.csv")))
    cur = list(csv.DictReader(open(N07 / "curve.csv")))

    check("N07 k_ref", s7["k_ref"], {"F0": 8, "C2": 7, "R": 0, "S": 0})
    check("N07 k_ref_effective", s7["k_ref_effective"], {"F0": 8, "C2": 7, "R": 7, "S": 7})

    def cell(st, tier, cond, arm, mode="matched"):
        return [r for r in c7 if r["stratum"] == st and r["tier"] == tier
                and r["condition"] == cond and r["mode"] == mode][0][arm]

    check("F0 all removed reader", cell("all", "F0", "tilt_removed", "reader"), "22/27")
    check("F0 all removed reader_start", cell("all", "F0", "tilt_removed", "dpvo_reader_start"), "13/27")
    check("F0 all mirror reader", cell("all", "F0", "mirrored_roll", "reader"), "15/27")
    check("F0 all scale reader", cell("all", "F0", "tilt_scaled", "reader"), "15/33")
    check("F0 forest feasible reader", cell("forest", "F0", "feasible", "reader"), "8/26")
    check("F0 forest removed reader", cell("forest", "F0", "tilt_removed", "reader"), "9/9")
    check("F0 forest removed reader_start", cell("forest", "F0", "tilt_removed", "dpvo_reader_start"), "4/9")
    check("F0 forest removed oracle", cell("forest", "F0", "tilt_removed", "dpvo_oracle"), "4/9")
    check("F0 forest feasible oracle", cell("forest", "F0", "feasible", "dpvo_oracle"), "2/26")
    check("F0 high removed reader_start", cell("high_mount", "F0", "tilt_removed", "dpvo_reader_start"), "13/13")
    check("F0 high feasible reader_start", cell("high_mount", "F0", "feasible", "dpvo_reader_start"), "8/40")
    check("F0 pooled10 forest removed 2deg", cell("forest", "F0", "tilt_removed", "dpvo_oracle_2deg", "10"), "9/9")
    check("F0 pooled10 forest feasible 2deg", cell("forest", "F0", "feasible", "dpvo_oracle_2deg", "10"), "2/26")
    check("F0 pooled10 forest feasible reader", cell("forest", "F0", "feasible", "reader", "10"), "10/26")
    check("F0 pooled10 forest mirror 2deg", cell("forest", "F0", "mirrored_roll", "dpvo_oracle_2deg", "10"), "9/9")
    check("F0 pooled10 forest mirror reader", cell("forest", "F0", "mirrored_roll", "reader", "10"), "6/9")
    check("C2 all removed reader", cell("all", "C2", "tilt_removed", "reader"), "16/27")
    check("C2 all removed reader_start", cell("all", "C2", "tilt_removed", "dpvo_reader_start"), "6/27")
    check("C2 all removed oracle", cell("all", "C2", "tilt_removed", "dpvo_oracle"), "6/27")
    check("C2 all removed level", cell("all", "C2", "tilt_removed", "dpvo_level"), "0/27")
    check("R all removed reader", cell("all", "R", "tilt_removed", "reader"), "10/27")
    check("R all removed reader_start", cell("all", "R", "tilt_removed", "dpvo_reader_start"), "6/27")
    check("S fixed0.5 feasible reader", s7["S_fixed_cut_0p5"]["reader"]["feasible"], "18/80")
    check("F0 tail >20 reader total", sum(s7["f0_tail_20deg"]["reader"].values()), 0)
    check("F0 tail >20 oracle 45", s7["f0_tail_20deg"]["dpvo_oracle"]["45.0"], 5)
    check("F0 tail >20 oracle 30", s7["f0_tail_20deg"]["dpvo_oracle"]["30.0"], 1)
    check("F0 natural all feasible reader_start", cell("all", "F0", "feasible", "dpvo_reader_start", "natural"), "5/80")
    check("F0 natural all removed reader_start", cell("all", "F0", "tilt_removed", "dpvo_reader_start", "natural"), "11/27")
    check("F0 natural forest feasible oracle", cell("forest", "F0", "feasible", "dpvo_oracle", "natural"), "1/26")
    check("F0 natural forest removed oracle", cell("forest", "F0", "tilt_removed", "dpvo_oracle", "natural"), "4/9")
    check("F0 matched threshold reader", round(s7["thresholds"]["F0"]["reader"], 2), 9.33)
    check("F0 matched threshold reader_start", round(s7["thresholds"]["F0"]["dpvo_reader_start"], 2), 17.98)
    check("F0 matched threshold oracle", round(s7["thresholds"]["F0"]["dpvo_oracle"], 2), 16.44)

    # forest curve: oracle 9/9 at 3/26, reader 9/9 at 3/26, reader_start 4/9
    def curve_cell(st, tier, arm, k, field):
        return [r for r in cur if r["stratum"] == st and r["tier"] == tier
                and r["arm"] == arm and int(r["k_feasible"]) == k][0][field]

    check("curve forest oracle 3/26 removed", curve_cell("forest", "F0", "dpvo_oracle", 3, "tilt_removed_det"), "9")
    check("curve forest reader_start 3/26 removed", curve_cell("forest", "F0", "dpvo_reader_start", 3, "tilt_removed_det"), "4")
    check("curve forest reader 0/26 removed", curve_cell("forest", "F0", "reader", 0, "tilt_removed_det"), "6")

    s8 = json.load(open(N08 / "summary.json"))
    check("N08 fail flagged", s8["n_fail_flagged"], 1)
    check("N08 c1", s8["criteria"]["c1_reduction"], False)
    check("N08 c2", s8["criteria"]["c2_no_harm"], True)
    check("N08 forest 45 agg", round(s8["criteria"]["c1_numbers"]["forest_45_aggregate"], 2), 8.65)
    check("N08 forest 45 mean singles", round(s8["criteria"]["c1_numbers"]["forest_45_mean_of_singles"], 2), 8.95)
    check("N08 forest 30 agg", round(s8["criteria"]["c1_numbers"]["forest_30_aggregate"], 2), 8.30)
    check("N08 forest 45 disagreement", round(s8["criteria"]["c3_forest_45_median_disagreement"], 1), 15.6)

    s9 = json.load(open(N09 / "summary.json"))
    check("N09 n clips", s9["n_clips"], 78)
    check("N09 rel 1s oracle", round(s9["rel"]["oracle"]["1s"], 2), 3.62)
    check("N09 rel 1s reader_start", round(s9["rel"]["readerstart"]["1s"], 2), 3.70)
    check("N09 rel 1s reader", round(s9["rel"]["reader"]["1s"], 2), 7.39)
    check("N09 abs median oracle", round(s9["abs_median_over_frames"]["oracle"], 2), 4.41)
    check("N09 abs median reader_start", round(s9["abs_median_over_frames"]["readerstart"], 2), 9.56)
    check("N09 abs median reader", round(s9["abs_median_over_frames"]["reader"], 2), 9.07)
    check("N09 abs 1s reader_start", round(s9["abs"]["readerstart"]["1s"], 2), 5.21)
    check("N09 abs 1s reader", round(s9["abs"]["reader"]["1s"], 2), 4.71)

    print("\n%d checks, %d failed" % (NCHECKS, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
