"""E6 pose-injection (generation complete; depicted-tilt scoring pending).

Assets:
  tab_e6_pose-injection_1.tex  LPIPS vs level/prior and prior coverage, per design
  fig_e6_pose-injection_1.pdf  LPIPS vs injected tilt, three maneuvers
  fig_e6_pose-injection_2.pdf  one contact sheet (level | injected), downscaled

Sources: track B results/anwm_injected/eval_injected_injected_true.json,
clips_injected_index.csv, sheets/C_turn_20_fixed_0_level_vs_injected.png.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

import common as C
import style

EVAL = "results/anwm_injected/eval_injected_injected_true.json"
INDEX = "results/anwm_injected/clips_injected_index.csv"
SHEET = "results/anwm_injected/sheets/C_turn_20_fixed_0_level_vs_injected.png"
ASSET = "e6_pose-injection"


def parse(cid):
    m = re.match(r"C_([a-z_]+?)_(\d+)_(course|fixed)_(\d+)$", cid)
    if m:
        return m.group(1), float(m.group(2))
    if cid.startswith("C_cruise"):
        return "cruise", 0.0
    return None, None


def build():
    ev = C.load_json("b", EVAL)
    rows = []
    for cid, d in ev.items():
        man, des = parse(cid)
        if man is None:
            continue
        rows.append(dict(cid=cid, maneuver=man, design=des,
                         lpips_level=d["lpips_vs_level_mean"],
                         lpips_prior=d["lpips_vs_prior_mean"],
                         cov_mean=d.get("prior_coverage_mean"),
                         cov_min=d.get("prior_coverage_min")))

    # ---- table: aggregate per maneuver/design
    groups = {}
    for r in rows:
        groups.setdefault((r["maneuver"], r["design"]), []).append(r)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Maneuver & design (deg) & clips & LPIPS vs level & LPIPS vs prior & prior coverage (mean / min) \\",
        r"\midrule",
    ]
    for (man, des) in sorted(groups, key=lambda k: (k[0], k[1])):
        g = groups[(man, des)]
        lines.append(f'{C.tex_escape(man)} & {des:.0f} & {len(g)} & '
                     f'{np.mean([r["lpips_level"] for r in g]):.3f} & '
                     f'{np.mean([r["lpips_prior"] for r in g]):.3f} & '
                     f'{np.mean([r["cov_mean"] for r in g]):.2f} / '
                     f'{np.min([r["cov_min"] for r in g]):.2f} \\\\')
    all_level = np.mean([r["lpips_level"] for r in rows])
    lines += [r"\midrule",
              f'Pooled & -- & {len(rows)} & {all_level:.3f} & '
              f'{np.mean([r["lpips_prior"] for r in rows]):.3f} & -- \\\\',
              r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- figure: LPIPS vs injected tilt
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.78)
    for i, man in enumerate(("turn", "climb_turn", "accel_brake")):
        sel = [r for r in rows if r["maneuver"] == man]
        xs = [r["design"] for r in sel]
        ys = [r["lpips_level"] for r in sel]
        ax.plot(xs, ys, style.MARKERS[i] + style.LINESTYLES[i], color=style.PALETTE[i],
                ms=3.2, label=man.replace("_", "-"))
    ax.set_xlabel("injected tilt (deg)")
    ax.set_ylabel("LPIPS vs paired level clip")
    ax.set_xlim(0, 22)
    ax.set_ylim(bottom=0)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e6_pose-injection_1.pdf")

    # ---- contact sheet (raster inside a vector page)
    img = plt.imread(C.src("b", SHEET))
    fig, ax = plt.subplots(figsize=(style.FULL, style.FULL * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(C.FIGURES / "fig_e6_pose-injection_2.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    C.write_csv(ASSET, [[r["cid"], r["maneuver"], r["design"],
                         f'{r["lpips_level"]:.6f}', f'{r["lpips_prior"]:.6f}',
                         f'{r["cov_mean"]:.6f}', f'{r["cov_min"]:.6f}'] for r in rows],
                ["clip_id", "maneuver", "design_deg", "lpips_vs_level",
                 "lpips_vs_prior", "prior_coverage_mean", "prior_coverage_min"])
    C.provenance(ASSET, [("b", EVAL), ("b", INDEX), ("b", SHEET)],
                 notes=("LPIPS means are stored per clip. Prior coverage is the stored "
                        "prior_coverage_mean/min on the injected arm. Depicted-tilt scoring "
                        "of the injected clips is pending; no verdict asset is built."),
                 derived=False)
    return dict(pooled=float(all_level), n=len(rows))


if __name__ == "__main__":
    out = build()
    print("E6 pooled LPIPS", round(out["pooled"], 3), "n", out["n"])
