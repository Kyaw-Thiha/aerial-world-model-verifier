"""E5 ANWM verdict, additional tables.

Assets:
  tab_e5_anwm-verdict_3.tex  positive controls: cruise noise, edit recovery, gate transfer
  tab_e5_anwm-verdict_4.tex  nuisance absorption and straight-cruise nulls
  tab_e5_anwm-verdict_5.tex  interpolation-free peak bounds
  tab_e5_anwm-verdict_6.tex  Regime A statistic comparison (sensitivity)

Sources: track A2 a2_anwm_score controls.json, summary.json, per_clip.json;
a2_parity/summary.json; a2_modes/modes.json nulls; a2_checks checks.json peak bounds;
report 08 §9 for the report-only comparison row.
"""
from __future__ import annotations

import numpy as np

import common as C

CONTROLS = "results/a2_anwm_score/20261109T203026Z_46a8a54/controls.json"
SUMMARY = "results/a2_anwm_score/20261109T203155Z_46a8a54/summary.json"
PERCLIP = "results/a2_anwm_score/20261109T203155Z_46a8a54/per_clip.json"
PARITY = "results/a2_parity/20261110T073039Z_b13f97d/summary.json"
MODES = "results/a2_modes/20261110T165549Z/modes.json"
CHECKS = "results/a2_checks/20261109T225321Z/checks.json"
REPORT08 = "reports/08_track_a2_anwm_scored.md"
ASSET = "e5_anwm-verdict"


def build():
    ctl = C.load_json("a2", CONTROLS)
    summ = C.load_json("a2", SUMMARY)
    pc = C.load_json("a2", PERCLIP)
    parity = C.load_json("a2", PARITY)["by_cell"]
    nulls = C.load_json("a2", MODES)["nulls"]
    peaks = C.load_json("a2", CHECKS)["peak_bounds_summary"]

    # ---- table 3: positive controls
    nf = ctl["noise_floor"]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Cruise clip & depicted median & p90 \\",
        r" & (deg) & (deg) \\",
        r"\midrule",
    ]
    for clip, d in nf["cruise_clips"].items():
        lines.append(f'{C.tex_escape(clip)} & {d["median_deg"]:.2f} & {d["p90_deg"]:.2f} \\\\')
    frames = []
    for clip in nf["cruise_clips"]:
        frames.extend(pc[clip]["depicted_per_frame"])
    frames = np.asarray(frames)
    lines += [r"\midrule",
              f'Pooled cruise frames ({len(frames)}) & {np.median(frames):.2f} & '
              f'p99 {np.percentile(frames, 99):.2f} \\\\',
              f'Released level frames & {nulls["gt_level_frames"]["median"]:.2f} & '
              f'p99 {nulls["gt_level_frames"]["p99"]:.2f} \\\\',
              r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Edit & applied (deg) & recovered (deg) & median abs.\ error (deg) \\",
        r"\midrule",
    ]
    for cond in ("roll_5", "roll_10", "roll_20", "pitch_5", "pitch_10", "pitch_20"):
        d = ctl["edit_recovery"]["conditions"][cond]
        axis, val = cond.split("_")
        lines.append(f'{axis} & {d["applied_deg"]:.0f} & '
                     f'{d["recovered_change_median_deg"]:.2f} & {d["median_abs_error_deg"]:.2f} \\\\')
    gt = ctl["gate_transfer"]
    lines += [r"\midrule",
              f'Gate transfer, released level & -- & {gt["n_dropped"]}/{gt["n_clips"]} dropped & -- \\\\',
              f'Gate transfer, generated clips & -- & '
              f'{summ["testable_counts"]["gate_dropped"]}/{summ["n_clips"]} dropped & -- \\\\',
              r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- table 4: nuisance absorption
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Cell & none / lag-only & full parity & absorbed & at bound & null median \\",
        r" & (deg) & (deg) & (deg) & (\%) & (deg) \\",
        r"\midrule",
    ]
    cells = [("spline_kd0.0_A", "A, drag-free"), ("spline_kd0.295_A", "A, drag 0.295"),
             ("spline_kd0.0_C", "C, drag-free"), ("spline_kd0.295_C", "C, drag 0.295")]
    for key, label in cells:
        d = parity[key]
        lines.append(f'{label} & {d["median_residual_lag_only"]:.2f} & '
                     f'{d["median_residual_parity"]:.2f} & {d["median_absorbed_deg"]:.2f} & '
                     f'{100*d["uptilt_at_bound_frac"]:.0f} & -- \\\\')
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\footnotesize Straight-cruise null (7 clips, median residual):} \\"]
    for key, label, kd in (("kd0.0_lag_only", "drag-free", "0"),
                           ("kd0.295_lag_only", "drag 0.295", "0.295")):
        n = nulls[key]
        lines.append(f'cruise, {label} & {n["cruise_median"]:.2f} & -- & -- & -- & '
                     f'p99 {n["cruise_p99"]:.2f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_5", "\n".join(lines))

    # ---- table 6: peak bounds
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"step (s) & $k_d$ & regime & peak $|a|$ & T/W lower bound & clips $>1/2/3$ \\",
        r" & & & (m s$^{-2}$) & & (\%) \\",
        r"\midrule",
    ]
    for dt in ("0.25", "0.667", "1.0"):
        for kd in ("0.0", "0.295"):
            for reg in ("A", "B", "C"):
                d = peaks[f"dt{dt}_kd{kd}_{reg}"]
                frac = (f'{100*d["frac_twr_gt_1"]:.0f} / {100*d["frac_twr_gt_2"]:.0f} / '
                        f'{100*d["frac_twr_gt_3"]:.0f}')
                lines.append(f'{dt} & {kd} & {reg} & {d["peak_accel_max_median"]:.1f} & '
                             f'{d["twr_lower_max_median"]:.2f} & {frac} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_6", "\n".join(lines))

    # ---- table 7: Regime A statistic comparison (report-only numbers, flagged)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Statistic & A testable & A infeasible & C testable & C infeasible \\",
        r"\midrule",
        r"spline median vs depicted (amendment 3) & 22 & 17 & 21--42 & 19--42 \\",
        r"frame-level median residual (amendment 5) & 2--9 & 2--9 & 11 & 11 \\",
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize Report-only comparison, transcribed from report 08 "
        r"\S9; the stored summary carries the amendment-5 statistic.} \\",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_7", "\n".join(lines))

    # ---- table 8: minimum consistent dt (censored at the designed step)
    md = summ["minimum_consistent_dt"]
    design = {}
    for cid, d in pc.items():
        design[cid] = (d.get("regime"), d.get("designed_tilt_deg"))
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Cell & $k_d$ & clips & median minimum dt (s) \\",
        r"\midrule",
    ]
    for kdkey, kd in (("kd_0.0", "0"), ("kd_0.295", "0.295")):
        cells = {}
        for e in md[kdkey]["per_clip"]:
            reg, des = design.get(e["clip_id"], (None, None))
            if reg == "A":
                label = "Regime A"
            elif reg == "C":
                label = f'C, {int(des)}$^\\circ$' if des in (5.0, 10.0, 20.0) else "C, other"
            else:
                label = None
            if label is None:
                continue
            cells.setdefault(label, []).append(e["dt_min_s"])
        for label in ("Regime A", "C, 5$^\\circ$", "C, 10$^\\circ$", "C, 20$^\\circ$"):
            vals = cells.get(label)
            if not vals:
                continue
            lines.append(f'{label} & {kd} & {len(vals)} & '
                         f'{float(np.median(vals)):.2f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_8", "\n".join(lines))
    C.provenance("e5_dtmin", [("a2", SUMMARY), ("a2", PERCLIP)],
                 notes=("Median per-clip minimum consistent dt from the stored summary, "
                        "grouped by regime and designed tilt; untestable clips are censored, "
                        "not dropped."),
                 derived=True)

    C.write_csv("e5_controls", [[clip, f'{d["median_deg"]:.4f}', f'{d["p90_deg"]:.4f}']
                                for clip, d in nf["cruise_clips"].items()],
                ["cruise_clip", "depicted_median_deg", "depicted_p90_deg"])
    C.provenance("e5_controls", [("a2", CONTROLS), ("a2", SUMMARY), ("a2", PERCLIP)],
                 notes=("Cruise per-clip medians/p90 stored; the pooled frame median/p99 is "
                        "derived by pooling the stored depicted_per_frame arrays. Edit "
                        "recovery and gate transfer are stored controls."),
                 derived=True)
    C.write_csv("e5_absorption", [
        [k,
         "--" if d["median_residual_lag_only"] is None else f'{d["median_residual_lag_only"]:.6f}',
         "--" if d["median_residual_parity"] is None else f'{d["median_residual_parity"]:.6f}',
         f'{d["median_absorbed_deg"]:.6f}', f'{d["uptilt_at_bound_frac"]:.6f}']
        for k, d in parity.items()],
        ["cell", "median_lag_only_deg", "median_parity_deg", "absorbed_deg", "uptilt_at_bound_frac"])
    C.write_csv("e5_peaks", [
        [key, f'{d["peak_accel_max_median"]:.6f}', f'{d["twr_lower_max_median"]:.6f}',
         f'{d["frac_twr_gt_1"]:.6f}', f'{d["frac_twr_gt_2"]:.6f}', f'{d["frac_twr_gt_3"]:.6f}']
        for key, d in peaks.items()],
        ["cell", "peak_accel_median", "twr_lower_median", "frac_gt1", "frac_gt2", "frac_gt3"])
    C.provenance("e5_absorption", [("a2", PARITY), ("a2", MODES)],
                 notes="Median absorbed and at-bound fractions from the parity summary.", derived=False)
    C.provenance("e5_peaks", [("a2", CHECKS)],
                 notes="Peak-bound summary from the stored checks file.", derived=False)
    return dict(peak_cells=len(peaks))


if __name__ == "__main__":
    print(build())
