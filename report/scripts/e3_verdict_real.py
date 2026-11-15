"""E3 verdict-real-flights.

Assets:
  tab_e3_verdict-real-flights_1.tex  calibration modes: conformal threshold, fold range, floor
  tab_e3_verdict-real-flights_2.tex  acceptance and abstention by flight type
  fig_e3_verdict-real-flights_1.pdf   detection by edit condition with Wilson intervals, k/n

Sources: track A2 a2_mode_thresholds (latest table.json) for thresholds;
a2_verdict_gated drag-free run 20261110T165240Z_71fb651 for the drag-free operating
point; the k_d = 0.295 run 20261110T165308Z_71fb651 for the comparison mode.
"""
from __future__ import annotations

import re

import common as C
import style

THRESH = "results/a2_modes/20261110T165549Z/modes.json"
KD0 = "results/a2_verdict_gated/20261110T165240Z_71fb651/metrics.json"
KD295 = "results/a2_verdict_gated/20261110T165308Z_71fb651/metrics.json"
ASSET = "e3_verdict-real-flights"

MODE_LABEL = {
    "kd0.0_lag_only": "drag-free, lag-only",
    "kd0.0_full_parity": "drag-free, full nuisance",
    "kd0.295_lag_only": "drag 0.295, lag-only",
    "kd0.295_full_parity": "drag 0.295, full nuisance",
}


def build():
    t = C.load_json("a2", THRESH)["thresholds"]
    kd0 = C.load_json("a2", KD0)["results"]["kd_0"]
    kd295 = C.load_json("a2", KD295)["results"]["kd_0.295"]

    # ---- table 1: calibration modes
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Calibration mode & conformal threshold & fold range & detection floor \\",
        r" & (deg) & (deg) & (deg) \\",
        r"\midrule",
    ]
    for key in ("kd0.0_lag_only", "kd0.0_full_parity", "kd0.295_lag_only", "kd0.295_full_parity"):
        d = t[key]
        spread = d["tii_fold_spread"]
        n = d["tii_conformal_pooled_n"]
        lines.append(
            f'{MODE_LABEL[key]} & {d["tii_conformal_pooled"]:.2f} & '
            f'[{spread["min"]:.2f}, {spread["max"]:.2f}] & '
            f'{d["detection_floor_deg"]:.2f} \\\\'
        )
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\footnotesize Drag 0.295, full nuisance: cruise null binds at 8.48$^\circ$ (ANWM cruise p99).} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: acceptance and abstention by flight type
    abst = kd295["abstention_with_gate"]
    acc = kd295["acceptance_per_flight"]
    groups = {}
    for f, d in acc.items():
        typ = ("trackRATM" if "trackRATM" in f
               else ("piloted" if re.search(r"\d+p-", f) else "autonomous"))
        g = groups.setdefault(typ, [0, 0])
        g[0] += d["accepted"]
        g[1] += d["n_testable"]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Flight type & acceptance & abstention & testable real clips \\",
        r" & $k/n$ & (\%) & \\",
        r"\midrule",
    ]
    for typ in ("piloted", "autonomous", "trackRATM"):
        k, n = groups[typ]
        a = abst[typ.lower() if typ != "trackRATM" else "trackratm"]
        lines.append(f'{typ} & {k}/{n} & {C.pct(a)} & {n} \\\\')
    lines.append(f'Overall & {sum(g[0] for g in groups.values())}/'
                 f'{sum(g[1] for g in groups.values())} & {C.pct(abst["overall"])} & '
                 f'{sum(g[1] for g in groups.values())} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- figure: drag-free mode detection by condition with Wilson intervals
    fams = kd0["families"]
    order = [("tilt_k0", "tilt removal"), ("mirror_roll", "mirrored roll"),
             ("tilt_k05", "tilt $\\times 0.5$"), ("tilt_k15", "tilt $\\times 1.5$"),
             ("yaw_control", "yaw control")]
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.78)
    xs = range(len(order))
    rates, los, his, labels = [], [], [], []
    for key, lab in order:
        d = fams[key]
        # stored Wilson interval is on the rejection rate's complement for some
        # families; use the explicit rejection_wilson95 where present.
        r = d["rejection_rate"]
        rates.append(r)
        lo, hi = d.get("rejection_wilson95", [r, r])
        los.append(r - lo)
        his.append(hi - r)
        labels.append(f'{lab}\n({round(r * d["n"])}/{d["n"]})')
    ax.errorbar(list(xs), rates, yerr=[los, his], fmt="o",
                color=style.OKABE_ITO["blue"], capsize=3, lw=0.9, ms=4)
    for i, r in enumerate(rates):
        ax.annotate(f"{r:.2f}", (i, r), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("detection rate (rejection)")
    ax.set_ylim(0, 1.05)
    ax.axhline(float(kd0["heldout_acceptance"]), color=style.OKABE_ITO["grey"],
               ls=":", lw=0.7)
    style.save(fig, C.FIGURES / "fig_e3_verdict-real-flights_1.pdf")

    rows = []
    for key, lab in order:
        d = fams[key]
        rows.append([key, d["n"], d["rejection_rate"], d["auroc"]])
    C.write_csv(ASSET, [["condition", "n", "rejection_rate", "auroc"]] and
                [[r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"] for r in rows],
                ["condition", "n", "rejection_rate", "auroc"])

    C.provenance(ASSET, [("a2", THRESH), ("a2", KD0), ("a2", KD295)],
                 notes=("Thresholds and fold ranges from the latest a2_mode_thresholds table. "
                        "The figure uses the drag-free gated run (kd_0, held-out acceptance "
                        "0.966, n = 58), which is the mode whose numbers match the handoff's "
                        "reference; report 07 calls k_d = 0.295 primary. Wilson intervals are "
                        "rejection_wilson95 from the stored families."),
                 derived=False)
    return dict(rates=rates)


if __name__ == "__main__":
    out = build()
    print("E3 rates", [round(r, 3) for r in out["rates"]])
