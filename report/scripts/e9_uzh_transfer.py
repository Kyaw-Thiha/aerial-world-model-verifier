"""E9 uzh-transfer.

Assets:
  tab_e9_uzh-transfer_1.tex  per-frame error, forward vs 45-down vs TII, by variant
  tab_e9_uzh-transfer_2.tex  mount: published elevation, fitted rotation, held-out residual
  tab_e9_uzh-transfer_3.tex  edit validity (null edit and known-rotation recovery)
  tab_e9_uzh-transfer_4.tex  verdict transfer on UZH forward rigs
  fig_e9_uzh-transfer_1.pdf  depicted-vs-truth lean slope per sequence

Sources: track E results/e_perframe/v1_forward/metrics.json, results/e_perframe/v1_all,
results/e_mount/mount.json, results/e_audit/audit.json,
results/e_verdict_v3/fin_370ms/metrics.json, results/e_calib/calibration.json,
track A phase1_depicted metrics for the TII reference.
"""
from __future__ import annotations

import numpy as np

import common as C
import style

PERF = "results/e_perframe/v1_forward/metrics.json"
PERF_ALL = "results/e_perframe/v1_all/metrics.json"
MOUNT = "results/e_mount/mount.json"
AUDIT = "results/e_audit/audit.json"
VERDICT = "results/e_verdict_v3/fin_370ms/metrics.json"
CALIB = "results/e_calib/calibration.json"
TII = "results/phase1_depicted/20261108T131825Z_c738aa8/metrics.json"
ASSET = "e9_uzh-transfer"


def build():
    perf = C.load_json("e", PERF)["variants"]
    perf_all = C.load_json("e", PERF_ALL)["variants"]
    audit = C.load_json("e", AUDIT)
    mount = C.load_json("e", MOUNT)["variants"]["b"]["per_rig"]
    verdict = C.load_json("e", VERDICT)
    calib = C.load_json("e", CALIB)
    tii = C.load_json("a", TII)

    # published elevation per rig (median over sequences)
    elev = {}
    for e in audit:
        rig = e["rig"]
        elev.setdefault(rig, []).append(e["mount"]["optical_axis_elevation_deg"])

    # ---- table 1: per-frame error
    def row(label, variant):
        p = variant["pooled"]
        return (f'{label} & {p["n"]} & {p["median_deg"]:.2f} & '
                f'{p["trimmed_rmse_10pct_deg"]:.2f} & {p["rmse_deg"]:.2f} & '
                f'{C.pct(p["frac_over_10_deg"])} \\\\')

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Data / variant & $n$ & median & trimmed RMSE & full RMSE & $>10^\circ$ \\",
        r" & & (deg) & (deg) & (deg) & (\%) \\",
        r"\midrule",
    ]
    pf = tii["pooled_all_frames"]
    lines.append(f'TII reference, raw fisheye & {pf["n"]} & {pf["median_deg"]:.2f} & '
                 f'{pf["trimmed_rmse_deg"]:.2f} & {pf["rmse_deg"]:.2f} & '
                 f'{C.pct(pf["frac_over_10_deg"])} \\\\')
    lines.append(r"\midrule")
    lines.append(row("UZH forward, (b) raw fisheye", perf["b"]))
    lines.append(row("UZH forward, (a) pinhole + focal prior", perf["a"]))
    lines.append(row("UZH forward, (c) pinhole, focal free", perf["c"]))
    lines.append(r"\midrule")
    lines.append(row("UZH 45$^\\circ$-down, (b)", perf_all["b"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: mount
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Rig & published elevation & fitted rotation & held-out, published & held-out, fitted \\",
        r" & (deg) & (deg) & (deg) & (deg) \\",
        r"\midrule",
    ]
    for rig in ("indoor_forward", "outdoor_forward", "indoor_45", "outdoor_45"):
        if rig not in mount:
            continue
        d = mount[rig]
        pub = float(np.median(elev.get(rig, [float("nan")])))
        rot = d.get("rotation_angle_deg")
        rp = d.get("heldout_residual_published_deg")
        rf = d.get("heldout_residual_fitted_deg")
        rot_s = "--" if rot is None else f"{rot:.2f}"
        rp_s = "--" if rp is None else f"{rp:.2f}"
        rf_s = "--" if rf is None else f"{rf:.2f}"
        lines.append(f'{rig.replace("_"," ")} & {pub:.1f} & {rot_s} & {rp_s} & {rf_s} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- table 3: edit validity
    vg = verdict["validity_gate"]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Sequence & pass rate \\",
        r"\midrule",
    ]
    for seq, d in vg["per_sequence"].items():
        lines.append(f'{C.tex_escape(seq)} & {d["pass_rate"]:.2f} ({d["n_pairs"]} pairs) \\\\')
    lines += [r"\midrule",
              f'All & {vg["n_pairs"]} pairs, overall pass = {str(vg["pass_all"]).lower()} \\\\',
              r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- table 4: verdict transfer
    pr = verdict["per_rig"]["indoor_forward"]
    conds = [("real", "real false reject"), ("null_edit", "null edit"),
             ("yaw_control", "yaw control"), ("tilt_removal", "tilt removal"),
             ("mirror_roll", "mirrored roll"), ("scale_0.5", "$\\times 0.5$"),
             ("scale_1.5", "$\\times 1.5$")]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & median residual & R2 band (TII) & coherence tier & $k/n$ (residual) \\",
        r" & (\%) & (\%) & (\%) & \\",
        r"\midrule",
    ]
    for key, label in conds:
        mr = pr["median_residual"][key]
        r2 = pr["r2"][key]
        co = pr["coherence"][key]
        lines.append(f'{label} & {100*mr["rate"]:.1f} & {100*r2["rate"]:.1f} & '
                     f'{100*co["rate"]:.1f} & {mr["k"]}/{mr["n"]} \\\\')
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize UZH forward rigs, 0.37 s window, 9 testable real clips, 4 sequences. "
              r"Rates are flag rates; for the real condition they are false rejects.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_4", "\n".join(lines))

    # ---- figure: lean slope per sequence
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.85)
    uzh = [(k, v) for k, v in calib["uzh"].items() if v.get("forward")]
    uzh.sort(key=lambda kv: kv[1]["slope"])
    names = [k.replace("_forward", "").replace("indoor", "in").replace("outdoor", "out")
             for k, _ in uzh]
    slopes = [v["slope"] for _, v in uzh]
    ax.bar(range(len(slopes)), slopes, color=style.OKABE_ITO["sky"], width=0.7)
    tii_slopes = [v["slope"] for v in calib["tii"].values()]
    ax.axhspan(min(tii_slopes), max(tii_slopes), color=style.OKABE_ITO["vermillion"],
               alpha=0.15, linewidth=0, label="TII flight range")
    ax.axhline(1.0, color=style.OKABE_ITO["black"], lw=0.6, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("depicted-vs-truth lean slope")
    ax.set_ylim(bottom=0)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e9_uzh-transfer_1.pdf")

    C.write_csv(ASSET, [[k, f'{v["slope"]:.4f}', f'{v["gx"]:.4f}', f'{v["gy"]:.4f}',
                         f'{v["corr"]:.4f}', f'{v["n"]}']
                        for k, v in calib["uzh"].items()],
                ["sequence", "slope", "gx", "gy", "corr", "n_frames"])
    C.provenance(ASSET, [("e", PERF), ("e", PERF_ALL), ("e", MOUNT), ("e", AUDIT),
                         ("e", VERDICT), ("e", CALIB), ("a", TII)],
                 notes=("Per-frame pooled values read directly. Published elevation is the "
                        "median optical-axis elevation in the UZH audit. Edit-validity pass "
                        "rates from validity_gate. Verdict transfer uses the 0.37 s headline "
                        "run. Slopes from e_calib/calibration.json."),
                 derived=True)
    return dict(n_seq=len(calib["uzh"]))


if __name__ == "__main__":
    print("E9 sequences", build()["n_seq"])
