"""E2 depicted-tilt-tii, additional tables.

Assets:
  tab_e2_depicted-tilt-tii_3.tex  attitude over time: per-flight fusion error + azimuth re-score
  tab_e2_depicted-tilt-tii_4.tex  six gate candidates with AUROC, coverage, kept/dropped error
  tab_e2_depicted-tilt-tii_5.tex  frozen gate and per-flight coverage / false drops

Sources: track A results/phase1b_fusion/20261108T133135Z_c738aa8/metrics.json,
results/phase3a_verdict/20261109T074134Z_492033f_lean/leans.json,
track A2 gate_validation.json (a2_gate/20261109T201449Z_298ff20).
"""
from __future__ import annotations

import common as C

FUSION = "results/phase1b_fusion/20261108T133135Z_c738aa8/metrics.json"
LEANS = "results/phase3a_verdict/20261109T074134Z_492033f_lean/leans.json"
GATE = "results/a2_gate/20261109T201449Z_298ff20/gate_validation.json"
ASSET = "e2_depicted-tilt-tii"


def build():
    fus = C.load_json("a", FUSION)
    leans = C.load_json("a", LEANS)
    gate = C.load_json("a2", GATE)["validation"]

    # ---- table 3: attitude over time
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Flight & tilt RMSE & thrust-axis error & lean error & fusion gain \\",
        r" & (deg) & (deg) & (deg) & \\",
        r"\midrule",
    ]
    for f, d in fus["leave_one_flight_out"].items():
        lines.append(f'{C.tex_escape(f)} & {d["tilt_rmse_deg"]:.2f} & '
                     f'{d["axis_vector_median_deg"]:.2f} & {d["lean_median_deg"]:.1f} & '
                     f'{d["gain"]:.1f} \\\\')
    lines += [r"\midrule",
              f'Pooled median flight & {fus["pooled"]["median_flight_tilt_rmse_deg"]:.2f} & -- & '
              f'{fus["pooled"]["median_flight_lean_deg"]:.1f} & -- \\\\']
    s = leans["summary"]
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\footnotesize Lean error with a fitted azimuth per window:} \\")
    for w, val in s["windowed_median_deg"].items():
        lines.append(f'{w} s window & -- & -- & {val:.1f} & -- \\\\')
    lines.append(f'one azimuth per clip & -- & -- & {s["one_per_clip_median_deg"]:.1f} & -- \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- table 4: gate candidates
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Signal & eligible & AUROC & 95\,\% interval & coverage & kept / dropped error \\",
        r" & & & & & (deg) \\",
        r"\midrule",
    ]
    names = {
        "S1_gravity_uncertainty": "S1 uncertainty",
        "S2_up_confidence": "S2 up confidence",
        "S3_latitude_confidence": "S3 latitude confidence",
        "S4_temporal_jitter": "S4 temporal jitter",
        "S5_vo_agreement": "S5 VO agreement",
        "S6_uncertainty_instability": "S6 uncertainty instability",
    }
    for key, label in names.items():
        d = gate["signals"][key]
        a = d["auroc"]
        elig = "yes" if d["eligible_to_freeze"] else "no"
        lines.append(f'{label} & {elig} & {a["auroc"]:.3f} & [{a["lo"]:.3f}, {a["hi"]:.3f}] & '
                     f'{d["lofo_coverage_overall"]:.2f} & {d["kept_median_error_deg"]:.2f} / '
                     f'{d["dropped_median_error_deg"]:.2f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- table 5: frozen gate + per-flight coverage
    sig = gate["signals"]["S3_latitude_confidence"]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Flight & coverage & good clips dropped & bad clips dropped \\",
        r" & (\%) & & \\",
        r"\midrule",
    ]
    for f, d in sig["lofo"].items():
        lines.append(f'{C.tex_escape(f)} & {100*d["coverage"]:.1f} & {d["n_dropped_neg"]} & '
                     f'{d["n_dropped_pos"]} \\\\')
    lines += [r"\midrule",
              f'Overall & {100*sig["lofo_coverage_overall"]:.1f} & -- & -- \\\\',
              r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_5", "\n".join(lines))

    # ---- data
    rows = []
    for f, d in fus["leave_one_flight_out"].items():
        rows.append([f, f'{d["tilt_rmse_deg"]:.4f}', f'{d["axis_vector_median_deg"]:.4f}',
                     f'{d["lean_median_deg"]:.4f}', d["gain"]])
    C.write_csv("e2_attitude_time", rows,
                ["flight", "tilt_rmse_deg", "axis_vector_error_deg", "lean_error_deg", "gain"])
    C.provenance("e2_attitude_time", [("a", FUSION), ("a", LEANS)],
                 notes=("Per-flight CHOSEN-gain values from leave_one_flight_out; the chosen "
                        "gain is 1.0 in every fold. Lean re-score from the separate azimuth "
                        "window run. Pooled values are medians over flights."),
                 derived=True)
    C.write_csv("e2_gate_candidates", [[k, gate["signals"][k]["eligible_to_freeze"],
                                        f'{gate["signals"][k]["auroc"]["auroc"]:.6f}',
                                        f'{gate["signals"][k]["lofo_coverage_overall"]:.6f}',
                                        f'{gate["signals"][k]["kept_median_error_deg"]:.4f}',
                                        f'{gate["signals"][k]["dropped_median_error_deg"]:.4f}']
                                       for k in names],
                ["signal", "eligible", "auroc", "coverage", "kept_median_deg", "dropped_median_deg"])
    C.provenance("e2_gate_candidates", [("a2", GATE)],
                 notes="Six pre-registered candidate signals and the frozen gate's validation.",
                 derived=False)
    return dict(n_flights=len(fus["leave_one_flight_out"]))


if __name__ == "__main__":
    print(build())
