"""E5 anwm-verdict.

Assets:
  fig_e5_anwm-verdict_1.pdf  depicted vs designed tilt for designed maneuvers
  tab_e5_anwm-verdict_1.tex  20-deg designs: residual, threshold, margin per mode/interpolant
  tab_e5_anwm-verdict_2.tex  infeasible/testable counts by cell and mode

Sources: track A2 a2_anwm_score per_clip.json (depicted tilt), track B clips_index.csv
(designs and required tilt), a2_modes/margins_regimeC_20deg.csv, a2_modes modes.json.
"""
from __future__ import annotations

import csv

import numpy as np

import common as C
import style

PERCLIP = "results/a2_anwm_score/20261109T203155Z_46a8a54/per_clip.json"
CLIPS = "results/anwm_clips/clips_index.csv"
MARGINS = "results/a2_modes/margins_regimeC_20deg.csv"
MODES = "results/a2_modes/20261110T165549Z/modes.json"
ASSET = "e5_anwm-verdict"

MODE_ORDER = ["drag-free lag-only", "drag-free full parity",
              "kd0.295 lag-only", "kd0.295 full parity"]
INFKEY = {
    "drag-free full parity": "inf_12.95",
    "kd0.295 lag-only": "inf_8.96",
    "kd0.295 full parity": "inf_8.48",
}


def build():
    clips = {}
    with open(C.src("b", CLIPS), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            clips[r["clip_id"]] = r
    pc = C.load_json("a2", PERCLIP)

    # ---- figure: C clips, depicted vs designed
    pts = []
    for cid, d in pc.items():
        if d.get("maneuver") in (None, "", "cruise"):
            continue
        try:
            des = float(d["designed_tilt_deg"])
        except (TypeError, ValueError):
            continue
        pts.append((d["maneuver"], des, d["depicted_median_deg"],
                    float(clips[cid]["required_tilt_1s_kd_deg"])))
    style.apply()
    fig, ax = style.new(style.HALF, ratio=0.78)
    for i, man in enumerate(("turn", "climb_turn", "accel_brake")):
        sel = [p for p in pts if p[0] == man]
        if not sel:
            continue
        xs = [p[1] for p in sel]
        ys = [p[2] for p in sel]
        ax.plot(xs, ys, "o", color=style.PALETTE[i], ms=3.2, label=man.replace("_", "-"))
    req = np.array([p[1] for p in pts])
    reqv = np.array([p[3] for p in pts])
    order = np.argsort(req)
    ax.plot(req[order], reqv[order], "-", color=style.OKABE_ITO["black"], lw=0.8,
            label="required tilt (drag 0.295)")
    if pts:
        A = np.vstack([np.array([p[1] for p in pts]), np.ones(len(pts))]).T
        slope, intercept = np.linalg.lstsq(A, np.array([p[2] for p in pts]), rcond=None)[0]
    else:
        slope, intercept = 0.0, 0.0
    ax.set_xlabel("designed tilt (deg)")
    ax.set_ylabel("depicted tilt (deg)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    style.legend_outside(ax)
    style.save(fig, C.FIGURES / "fig_e5_anwm-verdict_1.pdf")

    # ---- table 1: 20-deg margins
    rows = []
    with open(C.src("a2", MARGINS), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Mode & interpolant & residual & threshold & margin & margin, strictest fold \\",
        r" & & (deg) & (deg) & (deg) & (deg) \\",
        r"\midrule",
    ]
    best = None
    for mode in MODE_ORDER:
        for interp in ("spline", "step"):
            sel = [r for r in rows if r["mode"] == mode and r["interpolant"] == interp
                   and r["testable"] == "True"]
            if not sel:
                continue
            res = float(np.median([float(r["residual_deg"]) for r in sel]))
            thr = float(sel[0]["threshold_pooled_deg"])
            mpool = min(float(r["margin_pooled_deg"]) for r in sel)
            mstric = min(float(r["margin_strictest_fold_deg"]) for r in sel)
            word = "spline" if interp == "spline" else "step"
            mstr = f"{mstric:.2f}"
            if best is None or mstric < best[0]:
                best = (mstric, mode, interp, mpool)
            lines.append(f'{mode.replace("kd0.295","$k_d$ 0.295").replace("drag-free","drag-free")} & '
                         f'{word} & {res:.2f} & {thr:.2f} & {mpool:.2f} & {mstr} \\\\')
    lines += [r"\bottomrule",
              f'\\multicolumn{{6}}{{l}}{{\\footnotesize Minimum margin '
              f'{best[0]:.2f}$^\\circ$ ({best[1]}, {best[2]} interpolant); '
              f'positive margins mean the violation is detectable.}} \\\\',
              r"\end{tabular}"]
    C.write_tex(ASSET + "_1", "\n".join(lines))

    # ---- table 2: infeasible/testable counts by cell and mode
    modes = C.load_json("a2", MODES)["verdicts"]
    cells = [("A_native", "Regime A"), ("C_10", "C, 10$^\\circ$"),
             ("C_20", "C, 20$^\\circ$")]
    keymap = {
        "drag-free lag-only": "kd0.0_lag_only",
        "drag-free full parity": "kd0.0_full_parity",
        "kd0.295 lag-only": "kd0.295_lag_only",
        "kd0.295 full parity": "kd0.295_full_parity",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Cell & mode & clips & infeasible (spline / step) & testable (spline / step) \\",
        r"\midrule",
    ]
    csv_rows = []
    for ckey, clabel in cells:
        for mode in MODE_ORDER:
            d = modes[f"{keymap[mode]}_{ckey}"]
            by = d["by_interpolant"]
            n = d["n_clips"]
            if mode == "drag-free lag-only":
                inf_s = "--"
                inf_t = "--"
            else:
                k = INFKEY[mode]
                inf_s = f'{by["spline"].get(k, 0)}/{by["spline"]["n_clips"]}'
                inf_t = f'{by["step"].get(k, 0)}/{by["step"]["n_clips"]}'
            # testable = clips that are not infeasible (no excitation abstentions stored here)
            test_s = f'{by["spline"]["n_clips"] - by["spline"].get(INFKEY.get(mode, "inf_6.46"), 0)}/{by["spline"]["n_clips"]}' if mode != "drag-free lag-only" else "--"
            test_t = f'{by["step"]["n_clips"] - by["step"].get(INFKEY.get(mode, "inf_6.46"), 0)}/{by["step"]["n_clips"]}' if mode != "drag-free lag-only" else "--"
            lines.append(f'{clabel} & {mode} & {n} & {inf_s} / {inf_t} & {test_s} / {test_t} \\\\')
            csv_rows.append([ckey, mode, n, inf_s, inf_t])
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize Infeasible = residual above the mode's threshold. "
              r"Counts are clip counts from the stored verdicts; excitation abstentions are not "
              r"separately recorded in this file.} \\",
              r"\end{tabular}"]
    C.write_tex(ASSET + "_2", "\n".join(lines))

    C.write_csv(ASSET, [[p[0], p[1], f"{p[2]:.4f}", f"{p[3]:.4f}"] for p in pts],
                ["maneuver", "designed_tilt_deg", "depicted_median_deg", "required_tilt_kd_deg"])
    C.provenance(ASSET, [("a2", PERCLIP), ("b", CLIPS), ("a2", MARGINS), ("a2", MODES)],
                 notes=("Depicted medians from per_clip.json; required tilt from the clip set's "
                        "required_tilt_1s_kd_deg. Margins from margins_regimeC_20deg.csv "
                        "(minimum over the 5 clips of the 20-deg designs). Counts from the "
                        "stored per-mode verdicts; the drag-free lag-only threshold (14.73 deg) "
                        "has no matching inf_* key, so its infeasible counts are left blank "
                        "rather than approximated. Fit slope derived by report script."),
                 derived=True)
    return dict(slope=float(slope), intercept=float(intercept), n=len(pts))


if __name__ == "__main__":
    out = build()
    print("E5 slope", round(out["slope"], 3), "n", out["n"])
