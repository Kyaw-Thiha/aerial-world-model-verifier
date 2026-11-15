"""E3 verdict-real-flights, additional tables.

Assets:
  tab_e3_verdict-real-flights_3.tex  first verdict vs pre-registered criteria + policies
  tab_e3_verdict-real-flights_4.tex  per-flight real scores (the five 49-80 deg clips)
  tab_e3_verdict-real-flights_5.tex  nuisance ablation, fitted values, yaw artifact floor
  tab_e3_verdict-real-flights_6.tex  gated verdict with drag as an axis
  tab_e3_verdict-real-flights_7.tex  population x threshold definitions

Sources: track A phase3a metrics + ablations + report 04; track A2 gated runs,
pair-gate run, gate validation.
"""
from __future__ import annotations

import re

import common as C

PRIMARY = "results/phase3a_verdict/20261109T090957Z_bc3a398/metrics.json"
ABL = "results/phase3a_verdict/20261109T091536Z_ad3952a_ablations/ablations.json"
REPORT04 = "reports/04_track_a_gate3_verdict.md"
KD0 = "results/a2_verdict_gated/20261110T165240Z_71fb651/metrics.json"
KD15_295 = "results/a2_verdict_gated/20261110T165308Z_71fb651/metrics.json"
PAIR = "results/a2_verdict_gated_pairgate/20261109T202002Z_9e1656d/metrics.json"
ASSET = "e3_verdict-real-flights"


def build():
    p = C.load_json("a", PRIMARY)
    ev = p["evaluation"]
    abl = C.load_json("a", ABL)
    kd0 = C.load_json("a2", KD0)["results"]["kd_0"]
    kd295 = C.load_json("a2", KD15_295)["results"]["kd_0.295"]
    kd015 = C.load_json("a2", KD15_295)["results"]["kd_0.15"]
    pair = C.load_json("a2", PAIR)["results"]["kd_0.295"]

    # ---- table 3: first verdict against criteria
    fam = ev["families"]
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Criterion & target & measured & verdict \\",
        r"\midrule",
        f'held-out acceptance & $\geq 0.95$ & {ev["heldout_acceptance"]:.3f} '
        f'({round(ev["heldout_acceptance"]*ev["heldout_n"])}/{ev["heldout_n"]}) & fail \\\\',
        f'AUROC, tilt removal & $>0.9$ & {fam["tilt_k0"]["auroc"]:.3f} & pass \\\\',
        f'AUROC, $\\times 0.5$ & $>0.9$ & {fam["tilt_k05"]["auroc"]:.3f} & fail \\\\',
        f'AUROC, $\\times 1.5$ & $>0.9$ & {fam["tilt_k15"]["auroc"]:.3f} & fail \\\\',
        f'pooled AUROC, tilt families & -- & {ev["family_auroc_pooled"]:.3f} & -- \\\\',
        f'Spearman, score vs injected tilt & $>0.8$ & '
        f'{ev["spearman_score_vs_injected_deg"]:.2f} & pass \\\\',
        f'min detectable violation at 80\,\\% rejection & a curve & none & fail \\\\',
        f'yaw-control rejection & $\\pm5$ pp of real & '
        f'{100*fam["yaw_control"]["rejection_rate"]:.1f}\\% & fail \\\\',
        r"\midrule",
        r"\multicolumn{4}{l}{\footnotesize Three threshold policies:} \\",
    ]
    for key, label in (("primary", "primary (non-trackRATM)"),
                       ("all_flights", "all flights"),
                       ("all_but_13a", "all but 13a")):
        if key == "primary":
            thr = ev["pooled_thresholds"]["pooled"]["threshold"]
            acc = ev["heldout_acceptance"]
        elif key == "all_flights":
            d = ev["alt_threshold_policy_all_flights"]
            thr = {"all_flights": 67.16, "all_but_13a": 9.37}[key]
            acc = d["heldout_acceptance"]
        else:
            d = ev["alt_threshold_policy_all_but_13a"]
            thr = {"all_flights": 67.16, "all_but_13a": 9.37}[key]
            acc = d["heldout_acceptance"]
        vac = " (vacuous)" if key == "all_flights" else ""
        lines.append(f'{label} & {thr:.2f}$^\\circ$ threshold & acceptance {acc:.3f}{vac} & -- \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- table 4: per-flight real scores (parsed from report 04)
    text = " ".join(open(C.src("a", REPORT04), encoding="utf-8").read().split())
    rows = []
    for m in re.finditer(
        r"\|\s*\|?\s*\**([0-9]{2}[ap]-[a-zA-Z]+)\** \| (\d+) \| "
        r"\**([0-9.]+)[°]\** \| \**([0-9.]+)[°]\** \| "
        r"\**([0-9.]+)[°]\** \| ([0-9]+) ?% \|", text):
        rows.append((m.group(1), int(m.group(2)), float(m.group(3)),
                     float(m.group(4)), float(m.group(5)), float(m.group(6))))
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Flight & testable & median score & p90 & max & abstained \\",
        r" & & (deg) & (deg) & (deg) & (\%) \\",
        r"\midrule",
    ]
    for f, n, med, p90, mx, ab in rows:
        mark = "$^\\dagger$" if mx > 40 else ""
        lines.append(f'{C.tex_escape(f)}{mark} & {n} & {med:.2f} & {p90:.2f} & {mx:.2f} & {ab:.0f} \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{6}{l}{\footnotesize $^\dagger$Flight carries at least one real clip "
              r"scoring above 40$^\circ$; five such clips across 07a and 13a set the p95.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- table 5: nuisance ablation
    conds = [("identity", "real"), ("tilt_k0", "tilt removal"), ("tilt_k05", "$\\times 0.5$"),
             ("tilt_k15", "$\\times 1.5$"), ("mirror_roll", "mirrored roll"),
             ("yaw_control", "yaw control")]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & both & no uptilt & no lag & neither & absorbed & fitted uptilt / lag \\",
        r" & (deg) & (deg) & (deg) & (deg) & (deg) & \\",
        r"\midrule",
    ]
    for key, label in conds:
        a = abl["absorption_table_deg"][key]
        pc = abl["modes"]["both"]["per_condition"][key]
        lines.append(f'{label} & {a["both"]:.2f} & {a["no_uptilt"]:.2f} & {a["no_lag"]:.2f} & '
                     f'{a["none"]:.2f} & {a["absorption_by_nuisances_deg"]:.2f} & '
                     f'{pc["uptilt_offset_median_deg"]:+.0f}$^\\circ$ / '
                     f'{pc["lag_median_ms"]:+.0f} ms \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_5", "\n".join(lines))

    # ---- table 6: yaw-control artifact floor per flight
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Flight & median abs. tilt change & p90 & correlation with rotation \\",
        r" & (deg) & (deg) & \\",
        r"\midrule",
    ]
    for f, d in ev["artifact_floor"].items():
        lines.append(f'{C.tex_escape(f)} & {d["median_abs_tilt_change_deg"]:.2f} & '
                     f'{d["p90_abs_tilt_change_deg"]:.2f} & {d["corr_with_rotation"]:.2f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_6", "\n".join(lines))

    # ---- table 6: gated verdict with drag as axis
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Quantity & $k_d=0.295$ & $k_d=0.15$ & $k_d=0$ \\",
        r"\midrule",
    ]
    for key, label in (("heldout_acceptance", "acceptance"),):
        lines.append(f'{label} & {kd295[key]:.3f} ({kd295["heldout_n"]}) & '
                     f'{kd015[key]:.3f} ({kd015["heldout_n"]}) & '
                     f'{kd0[key]:.3f} ({kd0["heldout_n"]}) \\\\')
    thr = lambda d: d["pooled_thresholds"]["pooled"]["threshold"]
    lines.append(f'conformal threshold & {thr(kd295):.2f} & {thr(kd015):.2f} & {thr(kd0):.2f} \\\\')
    lines.append(f'pooled AUROC (unmatched) & {kd295["unmatched_auroc"]["pooled_tilt_families"]:.3f} & '
                 f'{kd015["unmatched_auroc"]["pooled_tilt_families"]:.3f} & '
                 f'{kd0["unmatched_auroc"]["pooled_tilt_families"]:.3f} \\\\')
    lines.append(r"\midrule")
    for key, label in (("tilt_k0", "removal"), ("tilt_k05", "$\\times 0.5$"),
                       ("tilt_k15", "$\\times 1.5$"), ("mirror_roll", "mirror"),
                       ("yaw_control", "yaw control")):
        lines.append(f'{label} rejected & {kd295["families"][key]["rejection_rate"]:.3f} & '
                     f'{kd015["families"][key]["rejection_rate"]:.3f} & '
                     f'{kd0["families"][key]["rejection_rate"]:.3f} \\\\')
    lines.append(r"\midrule")
    lines.append(f'gate drop, real & {kd295["abstention_with_gate"]["gate"]:.2f} & '
                 f'{kd015["abstention_with_gate"]["gate"]:.2f} & '
                 f'{kd0["abstention_with_gate"]["gate"]:.2f} \\\\')
    lines.append(f'gate drop, removal & -- & -- & -- \\\\')
    lines.append(r"\midrule")
    lines.append(f'pair-level removal & {pair["families"]["tilt_k0"]["rejection_rate"]:.3f} & -- & -- \\\\')
    lines.append(f'pair-level $\times 0.5$ & {pair["families"]["tilt_k05"]["rejection_rate"]:.3f} & -- & -- \\\\')
    lines.append(f'pair-level yaw & {pair["families"]["yaw_control"]["rejection_rate"]:.3f} & -- & -- \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_7", "\n".join(lines))

    # ---- table 8: definitions
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Population & threshold & behaviour & status \\",
        r"\midrule",
        r"testable pairs (per condition) & held-out LOFO conformal & pre-registered primary & primary \\",
        r"testable pairs (per condition) & fixed 6.46$^\circ$ & cross-drag comparison & withdrawn \\",
        r"gate-only pairs (gate on real, inherited) & held-out LOFO conformal & analysis, not pre-registered & sensitivity \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize The fixed-threshold column mixed a drag-0.295 "
        r"threshold with other modes and is withdrawn; see corrections.csv.} \\",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_8", "\n".join(lines))

    C.write_csv("e3_criteria", [
        ["held-out acceptance", f'{ev["heldout_acceptance"]:.6f}', ev["heldout_n"]],
        ["AUROC tilt removal", f'{fam["tilt_k0"]["auroc"]:.6f}', fam["tilt_k0"]["n"]],
        ["AUROC x0.5", f'{fam["tilt_k05"]["auroc"]:.6f}', fam["tilt_k05"]["n"]],
        ["AUROC x1.5", f'{fam["tilt_k15"]["auroc"]:.6f}', fam["tilt_k15"]["n"]],
        ["pooled AUROC tilt", f'{ev["family_auroc_pooled"]:.6f}', ""],
        ["Spearman", f'{ev["spearman_score_vs_injected_deg"]:.6f}', ""],
        ["primary threshold", f'{ev["pooled_thresholds"]["pooled"]["threshold"]:.6f}', ""],
    ], ["quantity", "value", "n"])
    C.write_csv("e3_nuisance", [
        [k, f'{v["both"]:.6f}', f'{v["no_uptilt"]:.6f}', f'{v["no_lag"]:.6f}',
         f'{v["none"]:.6f}', f'{v["absorption_by_nuisances_deg"]:.6f}']
        for k, v in abl["absorption_table_deg"].items()],
        ["condition", "both", "no_uptilt", "no_lag", "neither", "absorbed"])
    C.write_csv("e3_gated", [
        [m, f'{d["heldout_acceptance"]:.6f}', f'{thr(d):.6f}',
         f'{d["unmatched_auroc"]["pooled_tilt_families"]:.6f}']
        for m, d in (("kd_0.295", kd295), ("kd_0.15", kd015), ("kd_0", kd0))],
        ["mode", "acceptance", "threshold_deg", "pooled_auroc_unmatched"])
    C.provenance("e3_criteria", [("a", PRIMARY), ("a", REPORT04)],
                 notes=("Criteria and policies from the phase3a evaluation block; per-flight "
                        "real scores parsed from report 04's table (scores are not stored "
                        "per clip in metrics.json; alternative-policy thresholds 67.16 and "
                        "9.37 deg are quoted from report 04 because the evaluation block stores "
                        "only their acceptance)."),
                 derived=True)
    C.provenance("e3_nuisance", [("a", ABL), ("a", PRIMARY)],
                 notes="Ablation medians, fitted nuisance values and yaw artifact floor.",
                 derived=False)
    C.provenance("e3_gated", [("a2", KD0), ("a2", KD15_295), ("a2", PAIR)],
                 notes=("Gated runs at k_d 0.295 and 0.15 (run 165308) and k_d 0 (run 165240); "
                        "pair-level from the gate-on-real analysis run."),
                 derived=False)
    return dict(n_real_flights=len(rows))


if __name__ == "__main__":
    print(build())
