"""E10 window-rule-and-prototype.

Assets:
  tab_e10_window-rule-and-prototype_1.tex  rule window per TII flight and UZH sequence at
                                            truth rate and clip rate, with samples
  tab_e10_window-rule-and-prototype_2.tex  UZH forward verdict at rule windows vs 0.37 s
  tab_e10_window-rule-and-prototype_3.tex  1 Hz 16-point synthetic cases
  tab_e10_window-rule-and-prototype_4.tex  validated / not-validated scope matrix
  fig_e10_window-rule-and-prototype_1.pdf  threshold stability vs calibration clips

Sources: track D results/d_window_rates/d01/table.csv; track E results/e_verdict_v3/rule
and fin_370ms; track D results/d_bandwidth/d01/metrics.json (band stability);
reports/D04_bandwidth_window.md (synthetic coarse-path cases, parsed);
docs/prototype_v0.1.md and reports/D04, E05 for the scope matrix.
"""
from __future__ import annotations

import csv
import re

import common as C
import style

RATES = "results/d_window_rates/d01/table.csv"
STAB = "results/d_bandwidth/d01/metrics.json"
RULE = "results/e_verdict_v3/rule/metrics.json"
HEAD = "results/e_verdict_v3/fin_370ms/metrics.json"
D04 = "reports/D04_bandwidth_window.md"
E05 = "reports/E05_window_rule.md"
PROTO = "docs/prototype_v0.1.md"
ASSET = "e10_window-rule-and-prototype"


def build():
    # ---- table 1: two-rate windows
    rows = list(csv.DictReader(open(C.src("d", RATES), newline="", encoding="utf-8")))
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Data & unit & $n$ & truth window & truth samples & clip window & clip samples & floor (truth / clip) \\",
        r" & & & (s) & & (s) & & \\",
        r"\midrule",
    ]
    csv_rows = []
    last = None
    for r in rows:
        if last == "TII" and r["dataset"] != "TII":
            lines.append(r"\midrule")
        last = r["dataset"]
        fl = "pass / fail" if r["floor_truth"] == "True" else "fail / fail"
        lines.append(f'{r["dataset"]} & {C.tex_escape(r["unit"])} & {r["n"]} & '
                     f'{float(r["truth_window_med_s"]):.3f} & {float(r["truth_samples_med"]):.1f} & '
                     f'{float(r["clip_window_med_s"]):.3f} & {float(r["clip_samples_med"]):.1f} & {fl} \\\\')
        csv_rows.append([r["dataset"], r["unit"], r["n"], r["truth_window_med_s"],
                         r["truth_samples_med"], r["clip_window_med_s"], r["clip_samples_med"]])
    lines += [r"\bottomrule",
              r"\multicolumn{8}{l}{\footnotesize TII truth rate 274.73 Hz; UZH truth rate 500 Hz; "
              r"clip rates 12 Hz (TII) and about 15 Hz (UZH).} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: rule windows vs 0.37 s
    rule = C.load_json("e", RULE)["per_rig"]["indoor_forward"]
    head = C.load_json("e", HEAD)["per_rig"]["indoor_forward"]
    conds = [("real", "real false reject"), ("null_edit", "null edit"),
             ("yaw_control", "yaw control"), ("tilt_removal", "tilt removal"),
             ("mirror_roll", "mirrored roll"), ("scale_0.5", "$\\times 0.5$"),
             ("scale_1.5", "$\\times 1.5$")]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & R2 (rule) & R2 (0.37 s) & coherence (rule) & coherence (0.37 s) \\",
        r" & (\%) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for key, label in conds:
        r2r = rule["r2"][key]; r2h = head["r2"][key]
        cor = rule["coherence"][key]; coh = head["coherence"][key]
        lines.append(f'{label} & {100*r2r["rate"]:.1f} ({r2r["k"]}/{r2r["n"]}) & '
                     f'{100*r2h["rate"]:.1f} ({r2h["k"]}/{r2h["n"]}) & '
                     f'{100*cor["rate"]:.1f} ({cor["k"]}/{cor["n"]}) & '
                     f'{100*coh["rate"]:.1f} ({coh["k"]}/{coh["n"]}) \\\\')
    mr = rule["median_residual"]
    lines.append(r"\midrule")
    lines.append(f'Median residual, real$^\\dagger$ & {100*mr["real"]["rate"]:.1f} '
                 f'({mr["real"]["k"]}/{mr["real"]["n"]}) & -- & -- & -- \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize $^\dagger$Median-residual population is wider "
              r"(18 clips) than the gain/coherence population (9); the dagger marks intervals "
              r"that are descriptive with 4-7 sequences.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- table 3: 1 Hz 16-point synthetic cases (parse D04 text)
    text = " ".join(open(C.src("d", D04), encoding="utf-8").read().split())
    m = re.search(r"Synthetic checks: smooth curve f95 ([0-9.]+) → window ([0-9.]+) s, "
                  r"([0-9.]+) samples, abstain \(([^)]+)\); slalom f95 ([0-9.]+) → "
                  r"([0-9.]+) s, ([0-9.]+) samples, abstain \(([^)]+)\); straight line "
                  r"([0-9.]+) samples; waypoint step ([0-9.]+) samples", text)
    cases = []
    if m:
        cases = [
            ("smooth curve", float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)),
            ("slalom", float(m.group(5)), float(m.group(6)), float(m.group(7)), m.group(8)),
            ("straight line", None, None, float(m.group(9)), "floor"),
            ("waypoint step", None, None, float(m.group(10)), "floor"),
        ]
    lines = [
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Case & f95 (Hz) & window (s) & samples & outcome \\",
        r"\midrule",
    ]
    for name, f95, w, ns, why in cases:
        lines.append(f'{name} & {"--" if f95 is None else f"{f95:.4f}"} & '
                     f'{"--" if w is None else f"{w:.2f}"} & {ns:.2f} & abstain ({why}) \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize All 1 Hz, 16-point cases abstain; values parsed "
              r"from D04 \S``Coarse commanded paths''. The coarse-path replacement rule is a "
              r"proposal only and is not used here.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- table 4: scope matrix
    scope = [
        ("TII piloted", "validated (57 clips)", "calibrated only (57)",
         "validated (57)"),
        ("TII autonomous", "calibrated only (4)", "not validated (4)",
         "not validated (4)"),
        ("TII trackRATM", "not validated (4)", "not validated (4)",
         "not validated (4)"),
        ("UZH forward, indoor", "validated (9 clips, 4 seq.)",
         "calibrated only (9)", "validated (9, 4 seq.)"),
        ("UZH forward, outdoor", "not validated", "not validated",
         "not validated"),
        ("UZH 45$^\\circ$-down", "out of scope", "out of scope", "out of scope"),
    ]
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Flight type / rig & median residual & gain tier & coherence tier \\",
        r"\midrule",
    ]
    for r in scope:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule",
              r"\multicolumn{4}{l}{\footnotesize Built from the latest reports (D04, E05) and the "
              r"prototype document; where they disagree the later report (D04) is followed. "
              r"Entries carry their evidence counts.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- figure: threshold stability
    stab = C.load_json("d", STAB)["band_stability_tii"]
    ns = [10, 15, 20, 25, 30, 40, 50]
    vals = [100 * stab[str(n)]["tau_rel_iqr"] for n in ns]
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.72)
    ax.plot(ns, vals, "o-", color=style.OKABE_ITO["blue"])
    for n, v in zip(ns, vals):
        ax.annotate(f"{v:.1f}", (n, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6)
    ax.set_xlabel("calibration clips")
    ax.set_ylabel(r"threshold relative IQR (\%)")
    ax.set_ylim(bottom=0)
    style.save(fig, C.FIGURES / "fig_e10_window-rule-and-prototype_1.pdf")

    C.write_csv(ASSET, [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in csv_rows],
                ["dataset", "unit", "n", "truth_window_s", "truth_samples",
                 "clip_window_s", "clip_samples"])
    C.provenance(ASSET, [("d", RATES), ("d", STAB), ("e", RULE), ("e", HEAD),
                         ("d", D04), ("e", E05), ("d", PROTO)],
                 notes=("Two-rate windows from the stored d_window_rates table. Rule-window "
                        "verdict from e_verdict_v3/rule; 0.37 s comparison from fin_370ms. "
                        "Synthetic 1 Hz cases parsed from D04 because they are not stored as a "
                        "result file. The scope matrix is qualitative and cites the later "
                        "reports; D04 supersedes the prototype's 0.07-0.12 s 'truth rate' claim."),
                 derived=True)
    return dict(cases=len(cases))


if __name__ == "__main__":
    print("E10 synthetic cases", build()["cases"])
