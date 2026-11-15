"""E8 gain-and-coherence-tests.

Assets:
  tab_e8_gain-and-coherence-tests_1.tex  matched false-reject comparison
  tab_e8_gain-and-coherence-tests_2.tex  gain decomposition by flight type
  tab_e8_gain-and-coherence-tests_3.tex  window sensitivity
  tab_e8_gain-and-coherence-tests_4.tex  coherence AUROC and detection
  tab_e8_gain-and-coherence-tests_5.tex  flight-clustered intervals
  fig_e8_gain-and-coherence-tests_1.pdf  detection vs synthetic gain scale (gain test)

Sources: track D results/d_eval/d02/metrics.json, results/d_decomposition/d01,
results/d_window/d01, results/d_coherence/d01, results/d_flights/d01/interval.json,
reports/D04_bandwidth_window.md (coherence cluster intervals and procedure).
The baseline/coherence synthetic-gain curves are not stored; see MISSING.md.
"""
from __future__ import annotations

import csv

import numpy as np

import common as C
import style

D02 = "results/d_eval/d02/metrics.json"
DECOMP = "results/d_decomposition/d01/decomposition.json"
WINDOW = "results/d_window/d01/metrics.json"
COH = "results/d_coherence/d01/metrics.json"
INTERVAL = "results/d_flights/d01/interval.json"
D04 = "reports/D04_bandwidth_window.md"
ASSET = "e8_gain-and-coherence-tests"


def build():
    d02 = C.load_json("d", D02)
    decomp = C.load_json("d", DECOMP)
    win = C.load_json("d", WINDOW)
    coh = C.load_json("d", COH)
    inter = C.load_json("d", INTERVAL)

    # ---- table 1: matched false reject
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Family & gain test & median-residual baseline \\",
        r" & (\%, FRR 5.3\%) & (\%, FRR 3.4\%) \\",
        r"\midrule",
    ]
    fam_label = {"tilt_removal": "tilt removal", "mirror_roll": "mirrored roll",
                 "scale_0.5": "$\\times 0.5$", "scale_1.5": "$\\times 1.5$"}
    curve = list(csv.DictReader(open(C.src("d", "results/d_eval/d02/operating_curves.csv"),
                                   newline="", encoding="utf-8")))
    base = min((r for r in curve if r["method"] == "baseline"),
               key=lambda r: abs(float(r["frr"]) - d02["matched_false_reject"]["tilt_removal"]["baseline_frr"]))
    gain = min((r for r in curve if r["method"] == "gain"),
               key=lambda r: abs(float(r["frr"]) - d02["matched_false_reject"]["tilt_removal"]["gain_frr"]))
    for fam in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5"):
        lines.append(f'{fam_label[fam]} & {100*float(gain["det_"+fam]):.1f} & '
                     f'{100*float(base["det_"+fam]):.1f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: decomposition
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Flight type & truth to required & estimated to truth & verdict gain & dense required lean \\",
        r" & (slope) & (slope) & (slope) & (slope) \\",
        r"\midrule",
    ]
    a = decomp["all"]
    lines.append(f'All clips ({decomp["n_clips"]}, {a["a_gain"]["n_flights"]} flights) & '
                 f'{a["a_gain"]["median"]:.3f} & {a["b_gain"]["median"]:.3f} & '
                 f'{a["c_v1_gain"]["median"]:.3f} & {a["c_v2_gain"]["median"]:.3f} \\\\')
    for typ, d in decomp["by_type"].items():
        lines.append(f'{typ} & {d["a_gain"]["median"]:.3f} & {d["b_gain"]["median"]:.3f} & '
                     f'{d["c_v1_gain"]["median"]:.3f} & {d["c_v2_gain"]["median"]:.3f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- table 3: window sensitivity
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"window (s) & samples & false reject & removal & mirror & $\times 0.5$ & $\times 1.5$ & abstention \\",
        r" & & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for w in ("0.100", "0.185", "0.370", "0.740"):
        d = win["windows"][w]
        r = d["rates_frozen_band"]["R2_axis_gains"]
        lines.append(f'{float(w):.3f} & {d["native_samples"]} & {100*r["frr"]:.1f} & '
                     f'{100*r["det_tilt_removal"]:.1f} & {100*r["det_mirror_roll"]:.1f} & '
                     f'{100*r["det_scale_0.5"]:.1f} & {100*r["det_scale_1.5"]:.1f} & '
                     f'{100*d["abstention_window"]:.1f} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- table 4: coherence
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & AUROC (correlation) & detection rate & $n$ \\",
        r"\midrule",
    ]
    for cond in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5", "yaw_control"):
        d = coh["auroc"][cond]
        op = coh["operating_point"].get(cond, {})
        det = op.get("det_correlation")
        det_s = "--" if det is None else f"{100*det:.1f}"
        lines.append(f'{fam_label.get(cond, cond.replace("_"," "))} & '
                     f'{d["auroc_correlation"]:.3f} & {det_s} & {d["n"]} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- table 5: clustered intervals
    # coherence: per-flight flags from D04 (01p 2/26, 07p 1/24), recomputed here
    counts = {"flight-01a-ellipse": (2, 0), "flight-01p-ellipse": (26, 2),
              "flight-04a-ellipse": (2, 0), "flight-07p-lemniscate": (24, 1),
              "flight-13p-trackRATM": (2, 0), "flight-16p-trackRATM": (1, 0)}
    rng = np.random.default_rng(1)
    names = list(counts)
    pooled, unw = [], []
    for _ in range(5000):
        draw = rng.choice(len(names), size=len(names), replace=True)
        n = sum(counts[names[i]][0] for i in draw)
        k = sum(counts[names[i]][1] for i in draw)
        pooled.append(k / n)
        rates = [counts[names[i]][1] / counts[names[i]][0] for i in draw]
        unw.append(np.mean(rates))
    pooled = np.array(pooled)
    unw = np.array(unw)
    coh_lo, coh_hi = np.percentile(pooled, [2.5, 97.5])
    unw_lo, unw_hi = np.percentile(unw, [2.5, 97.5])

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Statistic & point & 95\,\% clustered interval & flags (flights) \\",
        r"\midrule",
        f'Gain tier, pooled & {100*inter["pooled_frr"]:.2f} \\% ({inter["n_false_rejects"]}/{inter["n_clips"]}) & '
        f'[{100*inter["pooled_frr_cluster95"][0]:.2f}, {100*inter["pooled_frr_cluster95"][1]:.2f}] & '
        f'{inter["n_false_rejects"]} ({inter["flights_with_false_reject"]} of 6) \\\\',
        f'Gain tier, unweighted mean & {100*inter["unweighted_mean_flight_rate"]:.2f} \\% & '
        f'[{100*inter["unweighted_cluster95"][0]:.2f}, {100*inter["unweighted_cluster95"][1]:.2f}] & -- \\\\',
        r"\midrule",
        f'Coherence, pooled$^\\dagger$ & 5.26 \\% (3/57) & [{100*coh_lo:.2f}, {100*coh_hi:.2f}] & 3 (2 of 6) \\\\',
        f'Coherence, unweighted mean$^\\dagger$ & 1.98 \\% & [{100*unw_lo:.2f}, {100*unw_hi:.2f}] & -- \\\\',
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize $^\dagger$Recomputed by report script from the "
        r"per-flight flags stated in D04, 5{,}000 replicates, \texttt{default\_rng(1)}; "
        r"descriptive with 6 clusters.} \\",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_5", "\n".join(lines))

    # ---- figure: gain test detection vs synthetic scale
    sweep = d02["synthetic_sweep"]
    ks = sorted({float(r["k"]) for r in sweep})
    det = []
    for k in ks:
        sel = [r for r in sweep if float(r["k"]) == k]
        det.append(sum(bool(r["flagged"]) for r in sel) / len(sel))
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.78)
    ax.plot(ks, det, "o-", color=style.OKABE_ITO["blue"], label="gain test")
    ax.set_xlabel("synthetic gain scale $k$")
    ax.set_ylabel("flag rate")
    ax.set_ylim(0, 1.04)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e8_gain-and-coherence-tests_1.pdf")

    C.write_csv(ASSET, [[k, f"{d:.6f}"] for k, d in zip(ks, det)],
                ["gain_scale", "gain_test_flag_rate"])
    C.provenance(ASSET, [("d", D02), ("d", DECOMP), ("d", WINDOW), ("d", COH),
                         ("d", INTERVAL), ("d", D04),
                         ("d", "results/d_eval/d02/operating_curves.csv")],
                 notes=("Matched false-reject and decomposition are stored. Window rates use "
                        "rates_frozen_band R2. Coherence AUROC/operating point stored. Gain "
                        "cluster intervals stored in d_flights; coherence cluster intervals "
                        "recomputed by the stated procedure from the per-flight flags in D04 "
                        "(the run's own interval file is not stored). Synthetic sweep stores "
                        "only the gain statistic; the baseline and coherence curves are not "
                        "stored and are omitted."),
                 derived=True)
    return dict(coh_lo=float(coh_lo), coh_hi=float(coh_hi))


if __name__ == "__main__":
    out = build()
    print("E8 coherence interval", round(100*out["coh_lo"], 2), round(100*out["coh_hi"], 2))
