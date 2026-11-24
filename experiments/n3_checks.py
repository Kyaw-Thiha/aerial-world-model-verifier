"""Track N3: machine checks that every headline number in the N3 reports is in the stored results.

Each check re-reads a stored JSON/CSV and compares it with the literal printed in the report. A
failure means a report sentence and the results disagree -- the run exits non-zero and says which.

Run:  PYTHONPATH=. pixi run python experiments/n3_checks.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED: list[tuple[str, bool, object]] = []


def check(label: str, got, want) -> None:
    ok = (abs(float(got) - float(want)) < 1e-6) if isinstance(want, (int, float)) else got == want
    CHECKED.append((label, bool(ok), got))


def main() -> int:
    a = json.loads((ROOT / "results/n3/n3a/summary.json").read_text())
    b = json.loads((ROOT / "results/n3/n3b/summary.json").read_text())
    c = json.loads((ROOT / "results/n3/n3c/summary.json").read_text())
    d = json.loads((ROOT / "results/n3/n3d/summary.json").read_text())

    # ------------------------------------------------------------------ N3A
    for tier, cov, dyn in (("P1", "9/200", "0/131"), ("P2", "25/200", "0/131"),
                           ("P3", "37/200", "2/131")):
        check(f"A sim {tier} coverage", a["overall"]["sim"]["tiers"][tier]["coverage"], cov)
        check(f"A sim {tier} dynamic coverage",
              a["overall"]["sim"]["tiers"][tier]["coverage_dynamic"], dyn)
    for tier, err, p90 in (("P1", 0.0, 0.073479), ("P2", 2.239946, 2.813213),
                           ("P3", 2.303269, 3.352761)):
        e = a["overall"]["sim"]["tiers"][tier]["anchor_err_deg"]
        check(f"A sim {tier} err median", round(e["median"], 6), round(err, 6))
        check(f"A sim {tier} err p90", round(e["p90"], 6), round(p90, 6))
    check("A forest feasible P1", a["criteria"]["availability_P1"]["forest_feasible"], "3/26")
    check("A availability gate fails", a["criteria"]["availability_P1"]["pass"], False)
    check("A accuracy gate passes", a["criteria"]["accuracy_P1"]["pass"], True)
    check("A uzh P1 raw", a["overall"]["uzh"]["tiers"]["P1"]["coverage_raw"], "12/14")
    check("A uzh P1 airborne", a["overall"]["uzh"]["tiers"]["P1"]["coverage"], "0/14")
    check("A tii P1 raw", a["overall"]["tii"]["tiers"]["P1"]["coverage_raw"], "6/8")
    check("A tii P1 airborne", a["overall"]["tii"]["tiers"]["P1"]["coverage"], "0/8")
    check("A uzh P3 airborne", a["overall"]["uzh"]["tiers"]["P3"]["coverage"], "1/14")
    surf = a["surface"]["sim"]["all"]
    for k, v in (("2deg_1s", "9/200"), ("4deg_1s", "25/200"), ("6deg_1s", "37/200"),
                 ("10deg_1s", "64/200"), ("12deg_1s", "83/200"), ("15deg_1s", "97/200")):
        check(f"A sim surface {k}", surf[k], v)
    for k, v in (("2deg_1s", "0/131"), ("6deg_1s", "2/131"), ("8deg_1s", "9/131")):
        check(f"A sim dynamic surface {k}", a["surface"]["sim"]["dynamic"][k], v)
    q = a["surface"]["sim"]["quietest_window_max_tilt_deg"]
    for k, v in (("1s_0.1", 3.55), ("1s_0.25", 8.64), ("1s_0.5", 15.38), ("1s_0.75", 30.92)):
        check(f"A quietest 1s {k}", round(q[k], 2), v)
    check("A hover coverage P1", a["sim_strata"]["P1"]["maneuver"]["hover"], "8/8")
    check("A ellipse coverage P1", a["sim_strata"]["P1"]["maneuver"]["ellipse"], "0/41")
    check("A diffpos P1", a["overall"]["sim_diffpos"]["tiers"]["P1"]["coverage"], "9/200")

    # the mount perturbation is exactly the perturbation size (checked on the raw rows)
    rows = list(csv.DictReader(open(ROOT / "results/n3/n3a/windows.csv")))
    m2 = [float(r["mount_err_2deg"]) for r in rows if r["mount_err_2deg"]]
    check("A mount err 2deg is 2.0", round(min(m2), 6), 2.0)
    check("A mount err 2deg is 2.0 (max)", round(max(m2), 6), 2.0)
    raw_uzh = [r for r in rows if r["kind"] == "uzh" and r["tier"] == "P1"
               and r["has_anchor_raw"] == "1"]
    check("A raw uzh anchors all pre-motion",
          all(r["raw_anchor_before_first_motion"] == "1" for r in raw_uzh), True)
    check("A raw uzh anchor count", len(raw_uzh), 12)

    # ------------------------------------------------------------------ N3B
    check("B reader F0 cross-check vs N07", round(max(b["config"]["cross_check_n07"].values()), 9),
          0.0)
    cov = b["coverage"]
    for arm, v in (("anchor1", "9/200"), ("anchor1_P2", "25/200"), ("anchor1_P3", "37/200")):
        check(f"B coverage {arm}", cov[arm]["all"], v)
    for cond, v in (("tilt_removed", "0/27"), ("mirrored_roll", "0/27")):
        check(f"B P1 scored {cond}", cov["anchor1"]["by_condition"][cond],
              "0/27" if v == "0/27" else v)
    check("B P3 anchored removals", cov["anchor1_P3"]["by_condition"]["tilt_removed"], "6/27")
    med = b["medians"]
    check("B reader feasible median", round(med["reader"]["feasible"], 2), 1.42)
    check("B reader_start feasible median", round(med["reader_start"]["feasible"], 2), 5.39)
    check("B oracle feasible median", round(med["oracle"]["feasible"], 2), 2.30)
    check("B anchor_P3 feasible median", round(med["anchor1_P3"]["feasible"], 2), 0.03)
    m = b["matched"]["F0"]
    for arm, v in (("reader", 9.33), ("reader_start", 17.98), ("oracle", 16.44),
                   ("anchor1_P3", 0.0)):
        check(f"B matched cut {arm}", round(m[arm]["threshold"], 2), v)
    for cond, v in (("tilt_removed", "22/27"), ("mirrored_roll", "15/27"), ("tilt_scaled", "15/33")):
        check(f"B reader matched {cond}", m["reader"][cond], v)
    for cond, v in (("tilt_removed", "6/27"), ("mirrored_roll", "6/27"), ("tilt_scaled", "6/33")):
        check(f"B anchor_P3 matched {cond}", m["anchor1_P3"][cond], v)
    f = b["matched_strata"]["F0"]["forest"]
    check("B forest reader removals", f["reader"]["tilt_removed"], "9/9")
    check("B forest reader feasible", f["reader"]["feasible"], "8/26")
    check("B forest reader_start removals", f["reader_start"]["tilt_removed"], "4/9")
    check("B forest anchor_P3 removals", f["anchor1_P3"]["tilt_removed"], "0/9")
    check("B forest anchor_P3 feasible", f["anchor1_P3"]["feasible"], "2/26")
    t = b["tail"]
    for arm, v in (("reader", 0), ("reader_start", 6), ("oracle", 6), ("anchor1_P3", 0)):
        check(f"B F0>20 tail {arm}", t[arm]["feasible_f0_gt_20"], v)
    op = b["anchored_subset_operating_point"]["anchor1_P3"]
    check("B subset cut", round(op["cut_deg"], 2), 2.54)
    for cond, v in (("feasible", "0/16"), ("tilt_removed", "6/6"), ("mirrored_roll", "5/6"),
                    ("tilt_scaled", "6/6"), ("partial", "2/3")):
        check(f"B subset {cond}", op[cond], v)
    cr = b["criteria"]
    check("B B1 fails", cr["B1_gate"]["n3a_availability_P1"].endswith("-> FAIL"), True)
    check("B B2 fails", cr["anchor1_P2"]["B2_forest"]["pass"], False)
    check("B B3 fails", cr["anchor1_P3"]["B3_overall"]["pass"], False)
    check("B B5 fails", cr["anchor1_P3"]["B5_vs_reader_start"]["pass"], False)

    # the six tail clips and all nine forest removals have no anchor at any tier
    per = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "results/n3/n3b/per_clip.csv"))}
    tail = ["i02_dev_003", "i02_dev_019", "i02_dev_038", "i02_dev_043", "i02_dev_059",
            "i02_dev_063"]
    def no_score(v: str) -> bool:
        return v in ("", "nan", "NaN")

    check("B tail clips have no anchor",
          all(no_score(per[x]["F0|anchor1_P3"]) for x in tail), True)
    forest_removed = [r["clip_id"] for r in per.values()
                      if r["family"] == "forest_ridge" and r["condition"] == "tilt_removed"]
    check("B forest removals count", len(forest_removed), 9)
    check("B forest removals have no anchor",
          all(no_score(per[x]["F0|anchor1_P3"]) for x in forest_removed), True)

    # ------------------------------------------------------------------ N3C
    check("C target is the fallback", c["target"]["name"], "oracle_tail_fallback")
    check("C target size", c["target"]["n"], 6)
    fl = c["criteria"]["all_flags"]
    check("C jit 10% recall", fl["jit|0.1"]["recall"], "3/6")
    check("C jit 10% precision", fl["jit|0.1"]["precision"], "3/20")
    check("C jit 20% recall", fl["jit|0.2"]["recall"], "4/6")
    check("C draw_dis_oracle 20% recall", fl["draw_dis_oracle|0.2"]["recall"], "5/6")
    check("C cover uninformative", fl["cover|0.1"]["recall"], "0/6")
    check("C rough uninformative", fl["rough|0.1"]["recall"], "0/6")
    check("C C1 fails", c["criteria"]["C1_recall"]["pass"], False)
    check("C C2 passes", c["criteria"]["C2_operating_point"]["pass"], True)
    check("C C3 passes", c["criteria"]["C3_forest_cost"]["pass"], True)
    a_jit = c["after_abstaining"]["oracle|jit"]
    check("C cut before", round(a_jit["cut_before_deg"], 2), 16.44)
    check("C cut after", round(a_jit["cut_after_deg"], 2), 10.92)
    check("C removals before", a_jit["before"]["tilt_removed"]["rate"], "13/27")
    check("C removals after", a_jit["after"]["tilt_removed"]["rate"], "15/24")
    check("C forest removals before", a_jit["before"]["forest_removed"]["rate"], "4/9")
    check("C forest removals after", a_jit["after"]["forest_removed"]["rate"], "6/7")
    check("C feasible after", a_jit["after"]["feasible"]["rate"], "8/74")
    dd = c["after_abstaining"]["oracle|draw_dis_oracle"]
    check("C two-draw forest after", dd["after"]["forest_removed"]["rate"], "8/8")

    # ------------------------------------------------------------------ N3D
    check("D P1 no anchor", d["P1"]["no_anchor_all"], "191/200")
    check("D P1 dynamic no anchor", d["P1"]["no_anchor_dynamic"], "131/131")
    check("D P3 no anchor", d["P3"]["no_anchor_all"], "163/200")
    check("D P3 dynamic no anchor", d["P3"]["no_anchor_dynamic"], "129/131")
    check("D P3 unreachable removals", d["P3"]["defects_unreachable"]["tilt_removed"], "21/27")

    # ------------------------------------------------------------------ report
    failed = [c for c in CHECKED if not c[1]]
    print(f"{len(CHECKED) - len(failed)}/{len(CHECKED)} checks match")
    for label, _, got in failed:
        print(f"  FAILED: {label} -> {got!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
