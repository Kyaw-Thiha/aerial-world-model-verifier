"""E2 depicted-tilt-tii.

Assets:
  tab_e2_depicted-tilt-tii_1.tex  per-flight and per-type per-frame error, adopted variant
  tab_e2_depicted-tilt-tii_1b.tex  estimator-route comparison (medians from report 02)
  fig_e2_depicted-tilt-tii_1.pdf   CDF of |tilt error| by track (derived from npz)
  fig_e2_depicted-tilt-tii_2.pdf   ROC of the frozen confidence gate, with permutation null

Sources: track A final phase1_depicted run (metrics.json + per-frame npz);
gate clip table from track A2 (a2_gate/20261109T201449Z_298ff20); permutation null
quoted from reports/06 (the null distribution itself is not stored as a file).
"""
from __future__ import annotations

import numpy as np

import common as C
import style

RUN = "results/phase1_depicted/20261108T131825Z_c738aa8"
ASSET = "e2_depicted-tilt-tii"
GATE = "results/a2_gate/20261109T201449Z_298ff20"
REPORT06 = "reports/06_track_a2_confidence_gate.md"


def roc(scores, labels):
    order = np.argsort(-np.asarray(scores, float))
    y = np.asarray(labels, int)[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (len(y) - y.sum())])
    return fpr, tpr


def build():
    m = C.load_json("a", f"{RUN}/metrics.json")
    rows = []
    for f, d in m["flights"].items():
        r = d["raw"]
        rows.append((f, d["track"], d["control"], r["n"], r["median_deg"],
                     r["trimmed_rmse_deg"], r["rmse_deg"], r["frac_over_10_deg"]))

    # by track (pooled over flights)
    by_track = {}
    for t in ("ellipse", "lemniscate", "trackRATM"):
        sel = [r for r in rows if r[1] == t]
        by_track[t] = (sum(r[3] for r in sel),)

    # ---- table 1: per flight
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Flight & track & $n$ & median & trimmed RMSE & full RMSE & $>10^\circ$ \\",
        r" & & & (deg) & (deg) & (deg) & (\%) \\",
        r"\midrule",
    ]
    for f, t, c, n, med, trm, rmse, frac in rows:
        lines.append(f'{C.tex_escape(f)} & {C.tex_escape(t)} & {n} & {med:.2f} & '
                     f'{trm:.2f} & {rmse:.2f} & {C.pct(frac)} \\\\')
    lines += [
        r"\midrule",
        f'Pooled (all frames) & -- & {m["pooled_all_frames"]["n"]} & '
        f'{m["pooled_all_frames"]["median_deg"]:.2f} & '
        f'{m["pooled_all_frames"]["trimmed_rmse_deg"]:.2f} & '
        f'{m["pooled_all_frames"]["rmse_deg"]:.2f} & '
        f'{C.pct(m["pooled_all_frames"]["frac_over_10_deg"])} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- variant comparison (report 02 survey)
    variants = [
        ("raw fisheye, \\texttt{simple\\_divisional}", 13.18, 192),
        ("pinhole, focal prior", 26.90, 48),
        ("pinhole, focal free", 25.41, 192),
    ]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Route & pooled median flight RMSE & frames \\",
        r" & (deg) & \\",
        r"\midrule",
    ]
    for name, med, n in variants:
        lines.append(f"{name} & {med:.2f} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- CDF figure (derived from per-frame npz)
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.72)
    xmax = 40.0
    for i, t in enumerate(("ellipse", "lemniscate", "trackRATM")):
        errs = []
        for f, d in m["flights"].items():
            if d["track"] != t:
                continue
            z = np.load(C.src("a", f"{RUN}/frames_{f}.npz"))
            errs.append(z["err_deg"])
        e = np.concatenate(errs)
        xs = np.sort(np.clip(e, 0, xmax))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=style.PALETTE[i], ls=style.LINESTYLES[i],
                label=f"{t} ($n$ = {len(xs)})")
    ax.set_xlabel(r"per-frame tilt error (deg)")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0.85, 1.001)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e2_depicted-tilt-tii_1.pdf")

    # ---- ROC figure from gate clip table
    import csv
    table = C.src("a2", f"{GATE}/clip_table.csv")
    with open(table, newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    sig = [float(r["identity::S3_latitude_confidence"]) for r in rd]
    lab = [int(r["label"]) for r in rd]
    fpr, tpr = roc(sig, lab)
    auroc = float(np.trapezoid(tpr, fpr))

    fig, ax = style.new(style.HALF, ratio=0.9)
    ax.step(fpr, tpr, where="post", color=style.OKABE_ITO["blue"],
            label=f"gate S3 (AUROC {auroc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color=style.OKABE_ITO["grey"], lw=0.7)
    # permutation null: median and 95th percentile of the best-of-four statistic
    ax.axhline(0.822, color=style.OKABE_ITO["vermillion"], lw=0.7, ls="--")
    ax.axhline(0.874, color=style.OKABE_ITO["orange"], lw=0.7, ls="-.")
    ax.text(0.02, 0.83, "null median 0.822", color=style.OKABE_ITO["vermillion"], fontsize=6)
    ax.text(0.02, 0.885, "null 95th pct 0.874", color=style.OKABE_ITO["orange"], fontsize=6)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style.legend_outside(ax, loc="lower right")
    style.save(fig, C.FIGURES / "fig_e2_depicted-tilt-tii_2.pdf")

    C.write_csv(ASSET, [[f, t, c, n, f"{med:.4f}", f"{trm:.4f}", f"{rmse:.4f}", f"{frac:.6f}"]
                        for f, t, c, n, med, trm, rmse, frac in rows],
                ["flight", "track", "control", "n_frames", "median_deg",
                 "trimmed_rmse_deg", "full_rmse_deg", "frac_over_10_deg"])

    C.provenance(ASSET, [("a", f"{RUN}/metrics.json"), ("a2", f"{GATE}/clip_table.csv"),
                         ("a", f"{RUN}/frames_flight-01p-ellipse.npz"), ("a2", REPORT06)],
                 notes=("Table rows from metrics.flights[*].raw; pooled from pooled_all_frames. "
                        "Trimmed RMSE is a 10% trim. CDF derived by report script from per-frame "
                        "err_deg arrays in the npz files. ROC derived by report script from the "
                        "gate clip table (S3 signal, label = median error > 10 deg); the "
                        "permutation-null medians (0.822, 0.874) are transcribed from report 06 "
                        "because the null distribution is not stored."),
                 derived=True)
    return dict(auroc=auroc)


if __name__ == "__main__":
    print("E2 gate AUROC", round(build()["auroc"], 4))
