"""E7 path-from-pixels, additional tables.

Assets:
  tab_e7_path-from-pixels_4.tex  implied tilt from the DPVO path on TII (two runs + sweep)
  tab_e7_path-from-pixels_5.tex  oracle ladder

Sources: track A phase2_implied runs 20261108T132738Z_c738aa8 (first) and
20261109T173302Z_d9de303 (corrected), phase1b_fusion for row 3.
"""
from __future__ import annotations

import common as C

FIRST = "results/phase2_implied/20261108T132738Z_c738aa8/metrics.json"
CORR = "results/phase2_implied/20261109T173302Z_d9de303/metrics.json"
FUSION = "results/phase1b_fusion/20261108T133135Z_c738aa8/metrics.json"
ASSET = "e7_path-from-pixels"


def build():
    first = C.load_json("a", FIRST)
    corr = C.load_json("a", CORR)
    fus = C.load_json("a", FUSION)

    # ---- table 4: two runs and corrected sweep
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Run / window & samples & implied-tilt RMSE & SNR \\",
        r" & & (deg) & \\",
        r"\midrule",
        f'first run (0.05 s, inert sweep) & -- & {first["pooled_by_window"]["0.05"]["median_flight_rmse_deg"]:.2f} & '
        f'{first["pooled_by_window"]["0.05"]["median_flight_snr"]:.2f} \\\\',
        f'corrected run, 5 samples & 5 & {corr["gate2"]["implied_tilt_rmse_deg"]:.2f} & '
        f'{corr["gate2"]["snr"]:.2f} \\\\',
        r"\midrule",
    ]
    for k, d in corr["pooled_by_window"].items():
        lines.append(f'corrected run, {k} samples & {k} & {d["median_flight_rmse_deg"]:.2f} & '
                     f'{d["median_flight_snr"]:.2f} \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{4}{l}{\footnotesize The first run's seconds-based sweep was inert "
              r"(all windows identical); only the corrected sample sweep is usable.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- table 5: oracle ladder
    first_l = first["oracle_ladder"]
    corr_l = corr["oracle_ladder"]
    vals = []
    for f, d in first["flights"].items():
        w = d["windows"].get("0.05", {})
        if "oracle_scale" in w and w["oracle_scale"].get("residual_median_deg") is not None:
            vals.append(w["oracle_scale"]["residual_median_deg"])
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Row & position & attitude & value (deg) \\",
        r"\midrule",
        r"1 & mocap (oracle) & mocap & "
        f'{first_l["mocap_position_mocap_attitude"]["median_rmse_deg"]:.2f} \\\\',
        r"2 & VO, oracle scale & mocap & "
        f'{first_l["vo_position_oracle_scale"]["median_rmse_deg"]:.2f} / '
        f'{corr_l["vo_position_oracle_scale"]["median_rmse_deg"]:.2f} (first / corrected) \\\\',
        r"3 & mocap & estimated (fusion) & "
        f'{fus["pooled"]["median_flight_tilt_rmse_deg"]:.2f} \\\\',
        r"4 & VO, oracle scale & estimated & "
        f'{min(vals):.2f}--{max(v for v in vals if v < 10):.2f} (self-consistency; 01p 26.5 excluded) \\\\',
        r"5 & fully estimated depth scale & -- & not run \\",
        r"6 & VO, fitted scale & -- & "
        f'{first_l["fully_estimated_fitted_scale"]["median_rmse_deg"]:.2f} / '
        f'{corr_l["fully_estimated_fitted_scale"]["median_rmse_deg"]:.2f} (first / corrected) \\\\',
        r"7 & commanded path & known & reference mode \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize Rows 4 is a consistency check, not accuracy: both "
        r"sides share the drifting VO frame. Rows 2 and 6 are accuracy.} \\",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_5", "\n".join(lines))

    C.write_csv("e7_implied", [[k, f'{d["median_flight_rmse_deg"]:.4f}',
                                f'{d["median_flight_snr"]:.4f}']
                               for k, d in corr["pooled_by_window"].items()],
                ["window_samples", "median_flight_rmse_deg", "median_flight_snr"])
    C.provenance("e7_implied", [("a", FIRST), ("a", CORR), ("a", FUSION)],
                 notes=("The two runs and the corrected sample sweep. The first run's "
                        "seconds-based sweep is inert and is not a usable curve. Oracle-ladder "
                        "row 4 is self-consistency; 01p's 26.5 deg residual is outside the "
                        "quoted 2.75-8.30 range and is noted."),
                 derived=True)
    return dict(ladder_rows=7)


if __name__ == "__main__":
    print(build())
