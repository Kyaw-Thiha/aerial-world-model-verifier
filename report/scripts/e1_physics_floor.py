"""E1 physics-floor: held-out residual vs speed (figure) and per-flight table.

Sources: track A phase0_floor metrics.json (final run 20261108T044153Z_f3b5d2bf).
Figure data: `residual_curves[flight].speed[bin]` medians for drag-free and fitted
drag, n-weighted across flights; the band is the min-max of per-flight medians.
Table: `loo_folds` per held-out flight.
"""
from __future__ import annotations

import numpy as np

import common as C
import style

SRC = "results/phase0_floor/20261108T044153Z_f3b5d2bf/metrics.json"
ASSET = "e1_physics-floor"


def load():
    return C.load_json("a", SRC)


def build():
    m = load()
    curves = m["residual_curves"]
    bins = list(next(iter(curves.values()))["speed"].keys())
    centers = [float(b.split("-")[0]) + 1.0 for b in bins]

    rows = []
    for b, c in zip(bins, centers):
        per_free, per_fit, ns = [], [], []
        for f, d in curves.items():
            cell = d["speed"].get(b)
            if not cell or cell["n"] == 0:
                continue
            per_free.append(cell["median_free"])
            per_fit.append(cell["median_fit"])
            ns.append(cell["n"])
        if not ns:
            continue
        w = np.asarray(ns, float)
        free = float(np.average(per_free, weights=w))
        fit = float(np.average(per_fit, weights=w))
        rows.append(
            dict(bin=b, center=c, n=int(sum(ns)), median_free=free,
                 median_fit=fit,
                 fit_min=float(min(per_fit)), fit_max=float(max(per_fit)),
                 free_min=float(min(per_free)), free_max=float(max(per_free)),
                 n_flights=len(ns))
        )

    C.write_csv(ASSET, [[r["bin"], r["center"], r["n"], r["n_flights"],
                         f'{r["median_free"]:.4f}', f'{r["median_fit"]:.4f}',
                         f'{r["free_min"]:.4f}', f'{r["free_max"]:.4f}',
                         f'{r["fit_min"]:.4f}', f'{r["fit_max"]:.4f}'] for r in rows],
                ["speed_bin_ms", "speed_center_ms", "n_frames", "n_flights",
                 "median_drag_free_deg", "median_fitted_drag_deg",
                 "drag_free_flight_min_deg", "drag_free_flight_max_deg",
                 "fitted_flight_min_deg", "fitted_flight_max_deg"])

    # pooled held-out medians
    folds = m["loo_folds"]
    pooled_free = float(m["gate"]["drag_free"]["median"])
    pooled_fit = float(m["gate"]["fitted_drag"]["median"])
    n_frames = int(m["gate"]["fitted_drag"]["n"])

    flights = []
    for f, d in folds.items():
        flights.append(dict(
            flight=f, window=d["window"], k_d=d["k_d"],
            median_fit=d["fitted_drag"]["median"], median_free=d["drag_free"]["median"],
            reduction=d["drag_free"]["median"] - d["fitted_drag"]["median"],
            p95=d["fitted_drag"]["p95"], n=d["n_heldout_frames"],
        ))

    # ---- figure
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.72)
    x = [r["center"] for r in rows]
    for key, lo, hi, col, ls, mk, label in [
        ("median_free", "free_min", "free_max", style.OKABE_ITO["vermillion"], "--", "s",
         "drag-free"),
        ("median_fit", "fit_min", "fit_max", style.OKABE_ITO["blue"], "-", "o", "fitted drag"),
    ]:
        y = [r[key] for r in rows]
        ax.fill_between(x, [r[lo] for r in rows], [r[hi] for r in rows],
                        color=col, alpha=0.16, linewidth=0)
        ax.plot(x, y, ls, color=col, marker=mk, label=label)
    ax.axhline(pooled_fit, color=style.OKABE_ITO["grey"], lw=0.6, ls=":")
    ax.set_xlabel("speed (m s$^{-1}$)")
    ax.set_ylabel("held-out residual (deg)")
    ax.set_xlim(0, 22)
    ax.set_ylim(bottom=0)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / f"fig_e1_physics-floor_1.pdf")

    # ---- table
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Flight & window & $k_d$ & median residual & drag reduction \\",
        r" & (samples) & (s$^{-1}$) & (deg) & (deg) \\",
        r"\midrule",
    ]
    for r in flights:
        lines.append(
            f'{C.tex_escape(r["flight"])} & {r["window"]} & {r["k_d"]:.3f} & '
            f'{r["median_fit"]:.2f} & {r["reduction"]:.2f} \\\\'
        )
    lines += [
        r"\midrule",
        f'Pooled (n = {n_frames:,}) & -- & -- & {pooled_fit:.2f} & '
        f'{pooled_free - pooled_fit:.2f} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    C.provenance(
        ASSET, [("a", SRC)],
        notes=("Per-flight values from loo_folds; speed curve from residual_curves, "
               "n-weighted across flights, band = min/max of per-flight medians. "
               "Pooled values are the stored pooled held-out medians (metrics.gate), "
               "not a weighted average of per-flight medians. "
               "Derived by report script: n-weighted aggregation and drag reduction."),
        derived=True,
    )
    return dict(pooled_fit=pooled_fit, pooled_free=pooled_free, flights=flights, rows=rows)


if __name__ == "__main__":
    out = build()
    print("E1 pooled fitted/free:", round(out["pooled_fit"], 3), round(out["pooled_free"], 3))
    for f in out["flights"]:
        print(f["flight"], round(f["median_fit"], 2), round(f["reduction"], 2))
