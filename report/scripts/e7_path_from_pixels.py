"""E7 path-from-pixels.

Assets:
  tab_e7_path-from-pixels_1.tex  extractor by sequence on UZH (direction error, magnitude, pass)
  tab_e7_path-from-pixels_2.tex  DPVO: calibrated vs estimated intrinsics
  tab_e7_path-from-pixels_3.tex  ANWM clips with DPVO and the failed verdict pilot
  fig_e7_path-from-pixels_1.pdf  direction error vs peak speed, marker = rig

Sources: track C results/pathrec_bakeoff/v2/summary.json,
results/pathrec_calib/focal_matrix.json, results/pathrec_calib/estimated_k.json,
results/pathrec_anwm/raw/anwmA_dpvo.json, results/pathrec_verdict/v1/summary.json,
track E results/e_perframe/v1_all/metrics.json (peak speed per sequence).
"""
from __future__ import annotations

import numpy as np

import common as C
import style

BAKEOFF = "results/pathrec_bakeoff/v2/summary.json"
FOCAL = "results/pathrec_calib/focal_matrix.json"
ESTK = "results/pathrec_calib/estimated_k.json"
ANWM = "results/pathrec_anwm/raw/anwmA_dpvo.json"
VERDICT = "results/pathrec_verdict/v1/summary.json"
PERF = "results/e_audit/audit.json"
ASSET = "e7_path-from-pixels"


def passes(d, ratio):
    return d <= 30.0 and 0.7 <= ratio <= 1.4


def build():
    summary = C.load_json("c", BAKEOFF)["table"]
    perf = {e["seq"]: e["kinematics"] for e in C.load_json("e", PERF)}

    # ---- table 1: extractor by sequence
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Extractor & sequence & direction error & magnitude ratio & pass \\",
        r" & & (deg) & & \\",
        r"\midrule",
    ]
    extractors = [("dpvo/UZH", "DPVO"), ("pi3/UZH", "Pi3"), ("vggt/UZH", "VGGT"),
                  ("da3/UZH", "DA3")]
    csv_rows = []
    for key, name in extractors:
        d = summary[key]
        npass = 0
        for c in d["per_clip"]:
            ok = c["pass"]
            npass += int(ok)
            lines.append(f'{name} & {C.tex_escape(c["seq"])} & {c["dir"]:.2f} & '
                         f'{c["ratio"]:.2f} & {"yes" if ok else "no"} \\\\')
            csv_rows.append([name, c["seq"], c["dir"], c["ratio"], ok])
        lines.append(f'\\multicolumn{{2}}{{l}}{{\\textit{{{name} total: {npass}/{d["n_clips"]}}}}} & '
                     f'{d["median_dir_err_deg"]:.2f} & {d["median_mag_ratio"]:.2f} & '
                     f'{npass}/{d["n_clips"]} \\\\')
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: intrinsics
    focal = C.load_json("c", FOCAL)
    est = C.load_json("c", ESTK)
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Intrinsics fed to DPVO & clips & median direction error (deg) & passing \\",
        r"\midrule",
    ]
    for factor in (0.8, 0.9, 1.0, 1.1, 1.2):
        sel = [e for e in focal if e.get("fx_scale") == factor]
        errs = [e["score"]["accel_dir_err_deg_median"] for e in sel]
        npass = sum(passes(e["score"]["accel_dir_err_deg_median"],
                           e["score"]["accel_mag_ratio"]) for e in sel)
        label = f"fx $\\times$ {factor:.1f}" + (" (calibrated)" if factor == 1.0 else "")
        lines.append(f'{label} & {len(sel)} & {np.median(errs):.2f} & {npass}/{len(sel)} \\\\')
    for cond in ("da3", "vggt"):
        sel = [e for e in est if e.get("condition") == cond]
        errs = [e["score"]["accel_dir_err_deg_median"] for e in sel]
        npass = sum(passes(e["score"]["accel_dir_err_deg_median"],
                           e["score"]["accel_mag_ratio"]) for e in sel)
        lines.append(f'{cond.upper()} estimated K & {len(sel)} & {np.median(errs):.2f} & '
                     f'{npass}/{len(sel)} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    # ---- ANWM DPVO counts
    anwm = C.load_json("c", ANWM)
    crash = degenerate = usable = 0
    for e in anwm:
        gen = e.get("generated")
        if gen is None:
            crash += 1
            continue
        if gen["accel_mag_ratio"] > 10.0:
            degenerate += 1
        else:
            usable += 1
    verdict = C.load_json("c", VERDICT)
    vreal = next(x for x in verdict if x["seq"] == "indoor_forward_5")
    vrem = next(x for x in verdict if x["seq"] == "indoor_forward_5_tilt_removed")
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"ANWM outcome (25 clips) & clips \\",
        r"\midrule",
        f'DPVO crashed & {crash} \\\\',
        f'degenerate trajectory (ratio $>10$) & {degenerate} \\\\',
        f'usable & {usable} \\\\',
        r"\midrule",
        r"Failed verdict pilot (UZH \texttt{indoor\_forward\_5}) & score (deg) \\",
        r"\midrule",
        f'real & {vreal["pix_score_deg"]:.1f} \\\\',
        f'tilt-removed & {vrem["pix_score_deg"]:.1f} \\\\',
        r"\bottomrule", r"\end{tabular}",
    ]
    C.write_tex(ASSET + "_3", "\n".join(lines))

    # ---- figure: direction error vs peak speed, marker = rig
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.78)
    for i, (rig, marker) in enumerate((("forward", "o"), ("45", "s"))):
        xs, ys = [], []
        for c in summary["dpvo/UZH"]["per_clip"]:
            seq = c["seq"]
            if rig == "forward" and "forward" not in seq:
                continue
            if rig == "45" and "forward" in seq:
                continue
            if seq not in perf:
                continue
            xs.append(perf[seq]["speed_p95_mps"])
            ys.append(c["dir"])
        ax.plot(xs, ys, marker, color=style.PALETTE[i], ms=3.4,
                label=f"{rig} rig" if rig == "forward" else "45$^\\circ$-down rig")
    ax.set_xlabel("peak speed (m s$^{-1}$)")
    ax.set_ylabel("direction error (deg)")
    ax.set_ylim(bottom=0)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e7_path-from-pixels_1.pdf")

    C.write_csv(ASSET, [[r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", r[4]] for r in csv_rows],
                ["extractor", "sequence", "direction_error_deg", "magnitude_ratio", "pass"])
    C.provenance(ASSET, [("c", BAKEOFF), ("c", FOCAL), ("c", ESTK), ("c", ANWM),
                         ("c", VERDICT), ("e", PERF)],
                 notes=("Sequence peak speeds from the UZH audit kinematics. Extractor rows and pass flags read from the v2 summary. Intrinsics "
                        "table counts the pre-registered pass criterion (direction <= 30 deg "
                        "and magnitude in [0.7, 1.4]) from the stored per-run scores. ANWM "
                        "counts derived by report script: crash = no generated block; "
                        "degenerate = accel magnitude ratio > 10. Verdict pilot scores read "
                        "from pathrec_verdict/v1."),
                 derived=True)
    return dict(crash=crash, degenerate=degenerate, usable=usable)


if __name__ == "__main__":
    out = build()
    print("E7 ANWM", out)
