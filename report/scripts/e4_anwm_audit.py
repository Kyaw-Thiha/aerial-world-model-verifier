"""E4 anwm-data-audit.

Assets:
  tab_e4_anwm-data-audit_1.tex  required tilt by regime and step duration (median, p90)
  tab_e4_anwm-data-audit_2.tex  fraction of steps above 10 deg
  tab_e4_anwm-data-audit_3.tex  designed maneuvers and unreachable designs
  tab_e4_anwm-data-audit_4.tex  level-frame depicted-tilt noise floor and seed variance

Sources: track B results/anwm_path_audit/audit.json (corrected pooled values),
results/anwm_clips/clips_index.csv (clip counts), results/anwm_clips/variance.json
(seed variance), track A anwm_floor_v1/summary.json (noise floor).
"""
from __future__ import annotations

import csv

import common as C

AUDIT = "results/anwm_path_audit/audit.json"
CLIPS = "results/anwm_clips/clips_index.csv"
VAR = "results/anwm_clips/variance.json"
FLOOR = "results/anwm_scoring/20261109T075519Z_f084f03/noise_floor.json"
ASSET = "e4_anwm-data-audit"

REGIME = {"A_native": "native paths", "B_rescaled": "rescaled racing",
          "C_designed": "designed maneuvers"}
DTS = ["0.250", "0.667", "1.000"]


def build():
    a = C.load_json("b", AUDIT)
    rows = []
    for reg in ("A_native", "B_rescaled", "C_designed"):
        cells = a[reg]["cells"]
        for dt in DTS:
            d = cells[dt]
            rows.append(dict(
                regime=REGIME[reg], dt=float(dt),
                ps_free=d["per_step_drag_free"], ps_kd=d["per_step_k_d"],
                sm_free=d["smoothed_drag_free"], sm_kd=d["smoothed_k_d"],
            ))

    def cell(d, stat):
        return f'{d["median"]:.2f} / {d["p90"]:.2f}'

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Regime & step (s) & per-step, free & per-step, drag & smoothed, free & smoothed, drag \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f'{r["regime"]} & {r["dt"]:.3f} & {cell(r["ps_free"],"m")} & '
                     f'{cell(r["ps_kd"],"m")} & {cell(r["sm_free"],"m")} & '
                     f'{cell(r["sm_kd"],"m")} \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{6}{l}{\footnotesize Entries are median / 90th percentile required tilt (deg).} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Regime & step (s) & per-step, free & per-step, drag & smoothed, free & smoothed, drag \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f'{r["regime"]} & {r["dt"]:.3f} & '
                     f'{C.pct(r["ps_free"]["frac_gt"])} & {C.pct(r["ps_kd"]["frac_gt"])} & '
                     f'{C.pct(r["sm_free"]["frac_gt"])} & {C.pct(r["sm_kd"]["frac_gt"])} \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{6}{l}{\footnotesize Fraction of steps with required tilt above $10^\circ$ (\%).} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # clip counts per (maneuver, policy, designed tilt)
    counts = {}
    with open(C.src("b", CLIPS), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                tgt = float(row["designed_tilt_deg"])
            except (TypeError, ValueError):
                tgt = None
            key = (row["maneuver"], row["policy"], tgt)
            counts[key] = counts.get(key, 0) + 1

    designs = a["C_designed"]["designs"]
    reach = [d for d in designs if d["reachable"]]
    unreach = [d for d in designs if not d["reachable"]]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Maneuver & yaw policy & target (deg) & clips & required, free & required, drag \\",
        r"\midrule",
    ]
    for d in reach:
        tgt = "--" if d["target"] is None else f'{d["target"]:.0f}'
        key = (d["maneuver"], d["policy"], None if d["target"] is None else float(d["target"]))
        n = counts.get(key, 0)
        lines.append(f'{C.tex_escape(d["maneuver"])} & {d["policy"]} & {tgt} & {n} & '
                     f'{d["achieved_1s_drag_free"]:.2f} & {d["achieved_1s_k_d"]:.2f} \\\\')
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\footnotesize Unreachable designs (at 1 s steps):} \\"]
    for d in unreach:
        tgt = "--" if d["target"] is None else f'{d["target"]:.0f}'
        lines.append(f'{C.tex_escape(d["maneuver"])} & {d["policy"]} & {tgt} & -- & '
                     f'{d["achieved_1s_drag_free"]:.2f} & {d["achieved_1s_k_d"]:.2f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    floor = C.load_json("a", FLOOR)["sizes"]
    var = C.load_json("b", VAR)
    seed_vals = [v["mean_pairwise_mae"] for v in var.values()]
    seed_n = sum(len(v["pairwise_mae"]) for v in var.values())
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Frames & $n$ & median (deg) & p90 (deg) & $>5^\circ$ (\%) \\",
        r"\midrule",
        f'512$\\times$512 & {floor["512"]["n"]} & {floor["512"]["median_deg"]:.2f} & '
        f'{floor["512"]["p90_deg"]:.2f} & {C.pct(floor["512"]["frac_gt_5deg"])} \\\\',
        f'224$\\times$224 & {floor["224"]["n"]} & {floor["224"]["median_deg"]:.2f} & '
        f'{floor["224"]["p90_deg"]:.2f} & {C.pct(floor["224"]["frac_gt_5deg"])} \\\\',
        r"\midrule",
        f'\\multicolumn{{5}}{{l}}{{Seed-to-seed pixel MAE: {sum(seed_vals)/len(seed_vals):.3f} '
        f'(mean of {len(var)} clips, {seed_n} pairs).}} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    C.write_csv(ASSET, [[r["regime"], f'{r["dt"]:.3f}',
                         f'{r["ps_free"]["median"]:.4f}', f'{r["ps_free"]["p90"]:.4f}',
                         f'{r["ps_kd"]["median"]:.4f}', f'{r["ps_kd"]["p90"]:.4f}',
                         f'{r["sm_free"]["median"]:.4f}', f'{r["sm_free"]["p90"]:.4f}',
                         f'{r["sm_kd"]["median"]:.4f}', f'{r["sm_kd"]["p90"]:.4f}',
                         f'{r["ps_free"]["frac_gt"]:.6f}', f'{r["ps_kd"]["frac_gt"]:.6f}',
                         f'{r["sm_free"]["frac_gt"]:.6f}', f'{r["sm_kd"]["frac_gt"]:.6f}']
                        for r in rows],
                ["regime", "step_s", "ps_free_median", "ps_free_p90", "ps_kd_median",
                 "ps_kd_p90", "sm_free_median", "sm_free_p90", "sm_kd_median", "sm_kd_p90",
                 "ps_free_frac_gt10", "ps_kd_frac_gt10", "sm_free_frac_gt10",
                 "sm_kd_frac_gt10"])

    C.provenance(ASSET, [("b", AUDIT), ("b", CLIPS), ("b", VAR), ("a", FLOOR)],
                 notes=("Regime statistics use the corrected pooled values in audit.json. "
                        "Clip counts are counted from clips_index.csv by maneuver/policy/design. "
                        "Seed variance is mean pairwise pixel MAE across 3 seeds. "
                        "Unreachable designs listed separately."),
                 derived=True)
    return dict(rows=rows, unreach=unreach)


if __name__ == "__main__":
    out = build()
    print("E4 rows", len(out["rows"]), "unreachable", len(out["unreach"]))
