"""H04: classify the 20 rejected arm-A clips (exploratory, post hoc, stated rule).

Every arm-A clip that is testable and rejected by the coherence tier is assigned one of the
three explanations the user asked for, by a rule fixed here before the table is read:

1. ``weak_signal`` — the frozen signal rules barely passed: the required tilt exceeds 10 deg on
   fewer than 30 % of frames (``signal_fraction < 0.30``) or the excitation is below 4 deg
   (twice the abstention floor).  The shape statistic has little to work with.
2. ``yaw_sensitive`` — not weak, and the same path's yaw-warped draw-0 clip is **not flagged**
   while this clip is: a tilt-preserving +20 deg view change alone clears it, so the rejection
   sits inside the perturbation band.
3. ``genuine_failure`` — otherwise.  Sub-mechanism from the per-axis correlations:
   ``roll_shape`` (r_y <= 0.30: the depicted bank does not follow the required bank),
   ``pitch_axis`` (r_y > 0.30 but r_x <= 0.50: the forward axis fails; that axis is the one the
   reader's fixed 90 deg focal prior makes uncertain), ``sign_tier`` (flagged by the sign
   sub-tier only, with r_min >= 0.6: the shape is right and the sign statistic is at chance),
   ``other``.

The yaw control's own score per path is reported beside every row, so a reader can check that
"yaw_sensitive: 0" is a measurement, not an assumption.

    PYTHONPATH=. pixi run -e gpu python experiments/h04_rejection_breakdown.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXCITATION_FLOOR_DEG = 4.0
SIGNAL_FRACTION_MIN = 0.30
TAU_CORR = 0.619653015661502
TAU_SIGN = 0.2921099290780138


def main() -> int:
    d = json.loads((ROOT / "results/h04/scored_main.json").read_text())
    rows = json.loads((ROOT / "results/h04/scored_main_rows.json").read_text())
    yaw = {p["path"]: p for p in d.get("yaw_control_pairs", [])}
    paths = np.load(ROOT / "results/h04/paths/paths.npz")
    sig = {r["path"]: float(np.nanmean(paths[f"{r['path']}__tilt_req_rule"] > 10.0))
           for r in rows}
    A = [r for r in rows if r["arm"] == "A_flatness"]
    rejected = [r for r in A if r["testable"] and not r["accepted"]]
    table = []
    for r in sorted(rejected, key=lambda x: (x["family"], x["path"], x["draw"])):
        sf = sig[r["path"]]
        yp = yaw.get(r["path"], {})
        yaw_flagged = yp.get("yaw_flagged")
        yaw_clears = bool(yaw_flagged is False)
        if sf < SIGNAL_FRACTION_MIN or r["excitation_deg"] < EXCITATION_FLOOR_DEG:
            cls, mech = "weak_signal", ""
        elif yaw_clears:
            cls, mech = "yaw_sensitive", ""
        else:
            cls = "genuine_failure"
            r_min = min(r["r_x"], r["r_y"])
            if r["r_y"] <= 0.30:
                mech = "roll_shape"
            elif r["r_x"] <= 0.50:
                mech = "pitch_axis"
            elif r["flagged_coh_sign"] and not r["flagged_coh_corr"] and r_min >= 0.6:
                mech = "sign_tier"
            else:
                mech = "other"
        table.append({"clip": r["clip_id"], "path": r["path"], "family": r["family"],
                      "peak": r["requested_peak"], "draw": r["draw"], "scene": r["scene"],
                      "excitation_deg": r["excitation_deg"], "signal_fraction": sf,
                      "r_x": r["r_x"], "r_y": r["r_y"], "s_x": r["s_x"], "s_y": r["s_y"],
                      "corr_score": r["corr_score"], "sign_score": r["sign_score"],
                      "flagged_by": ("corr" if r["flagged_coh_corr"] else "") +
                                    ("sign" if r["flagged_coh_sign"] else ""),
                      "yaw_counterpart_flagged": yaw_flagged,
                      "yaw_corr_score": yp.get("yaw_corr"),
                      "main_corr_score": yp.get("main_corr"),
                      "class": cls, "mechanism": mech})
    counts = {k: sum(t["class"] == k for t in table)
              for k in ("weak_signal", "yaw_sensitive", "genuine_failure")}
    mech = {k: sum(t["mechanism"] == k for t in table)
            for k in ("roll_shape", "pitch_axis", "sign_tier", "other")}
    out = {"n_armA": len(A), "n_testable": sum(r["testable"] for r in A),
           "n_accepted": sum(r["accepted"] for r in A), "n_rejected": len(rejected),
           "counts": counts, "mechanisms": mech,
           "rule": {"excitation_floor_deg": EXCITATION_FLOOR_DEG,
                    "signal_fraction_min": SIGNAL_FRACTION_MIN,
                    "tau_corr": TAU_CORR, "tau_sign": TAU_SIGN,
                    "note": "post hoc, exploratory; the rule was written before the table was read"},
           "rows": table}
    (ROOT / "results/h04/rejection_breakdown.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_armA", "n_testable", "n_accepted",
                                          "n_rejected", "counts", "mechanisms")},
                     indent=1))
    for t in table:
        print(f"{t['clip'][:40]:40s} {t['class']:16s} {t['mechanism']:10s} "
              f"exc {t['excitation_deg']:5.2f} sig {t['signal_fraction']:.2f} "
              f"rx {t['r_x']:5.2f} ry {t['r_y']:5.2f} flag {t['flagged_by']:4s} "
              f"yaw {t['yaw_counterpart_flagged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
