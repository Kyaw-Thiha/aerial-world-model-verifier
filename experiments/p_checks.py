"""Track P: machine checks of every headline number in the three reports against stored results.

    PYTHONPATH=. pixi run python experiments/p_checks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P0 = ROOT / "results/p/p0"
P1 = ROOT / "results/p/p1"
P2 = ROOT / "results/p/p2"
FAILS = []
N = 0


def check(name: str, value, expected, tol: float = 0.0) -> None:
    global N
    N += 1
    ok = (value == expected) if tol == 0 else (abs(float(value) - float(expected)) <= tol)
    if not ok:
        FAILS.append(f"{name}: got {value!r}, expected {expected!r} (±{tol})")
    print(("ok   " if ok else "FAIL ") + name + f" = {value!r}")


def main() -> None:
    s0 = json.load(open(P0 / "summary.json"))
    stop = s0["stop_rule"]
    check("P0 stop population", stop["population"], 53)
    check("P0 stop not-detectable", stop["not_detectable"], 11)
    check("P0 stop verdict", stop["stop"], False)
    check("P0 stop by_family removal", stop["by_family"]["tilt_removed"], "16/16")
    check("P0 stop by_family mirror", stop["by_family"]["mirrored_roll"], "5/16")
    check("P0 stop by_family scaled", stop["by_family"]["tilt_scaled"], "21/21")
    check("P0 filter cutoff", s0["filter"]["cutoff_3db_hz"], 4.048, 0.01)
    check("P0 sim step error", s0["noise"]["sim"]["step_err_median_deg"], 0.1337, 1e-4)
    check("P0 uzh step error", s0["noise"]["uzh"]["step_err_median_deg"], 0.1180, 1e-4)
    check("P0 tii step error", s0["noise"]["tii"]["step_err_median_deg"], 2.5587, 1e-4)
    check("P0 sim MDR", s0["budget_by_kind_window"]["sim_w0.5"]["mdr_window_deg_s"], 2.997, 0.01)
    check("P0 uzh MDR", s0["budget_by_kind_window"]["uzh_w0.5"]["mdr_window_deg_s"], 1.757, 0.01)
    check("P0 tii MDR", s0["budget_by_kind_window"]["tii_w0.5"]["mdr_window_deg_s"], 60.14, 0.1)
    check("P0 sim self-consistency", s0["budget_by_kind_window"]["sim_w0.5"]
          ["self_consistency_med_deg_s"], 0.00085, 2e-5)
    check("P0 sim literal", s0["budget_by_kind_window"]["sim_w0.5"]["literal_med_deg_s"],
          8.7389, 0.01)
    check("P0 sim required median", s0["budget_by_kind_window"]["sim_w0.5"]["req_med_deg_s"],
          9.0645, 0.01)
    check("P0 uzh required median", s0["budget_by_kind_window"]["uzh_w0.5"]["req_med_deg_s"],
          29.03, 0.1)
    check("P0 scale 0.1 detected", s0["scale_sweep"]["detectable_by_abs_scale_error"]["0.1"],
          "25/106")
    check("P0 scale 0.5 detected", s0["scale_sweep"]["detectable_by_abs_scale_error"]["0.5"],
          "96/106")
    bud = pd.read_csv(P0 / "budget.csv")
    w = bud[bud.window_s == 0.5]
    check("P0 sim required p90", np.percentile(w[w.kind == "sim"].req_med_deg_s, 90) > 42, True)

    s1 = json.load(open(P1 / "summary.json"))
    check("P1 abs threshold 4/80", s1["thresholds_sim"]["abs"]["4"], 40.175, 0.01)
    check("P1 cos threshold 4/80", s1["thresholds_sim"]["cos"]["4"], 0.1909, 1e-4)
    b = s1["sim_detection_at_4of80"]
    check("P1 feasible flags", b["feasible"]["abs"], "4/57")
    check("P1 removal flags", b["removal"]["abs"], "2/19")
    check("P1 mirror flags", b["mirror"]["abs"], "1/16")
    check("P1 scale flags", b["scale"]["abs"], "0/22")
    check("P1 removal coherence", b["removal"]["cos"], "14/16")
    check("P1 mirror coherence", b["mirror"]["cos"], "0/16")
    check("P1 partial flags", b["partial"]["abs"], "0/12")
    check("P1 yaw control", s1["yaw_control"]["flagged"], "4/57")
    check("P1 null edit", s1["null_edit"]["flagged"], "2/57")
    loo = s1["uzh_leave_one_out"]
    check("P1 uzh of1 FR", loo["outdoor_forward_1"]["window_false_reject"], 0.0, 1e-12)
    check("P1 uzh of5 FR", loo["outdoor_forward_5"]["window_false_reject"], 0.0185, 1e-3)
    check("P1 uzh 45 FR", loo["outdoor_45_1"]["window_false_reject"], 0.9310, 1e-3)
    check("P1 uzh of1 clip score", loo["outdoor_forward_1"]["clip_score_deg_s"], 3.6255, 1e-3)
    check("P1 uzh 45 clip score", loo["outdoor_45_1"]["clip_score_deg_s"], 25.715, 1e-2)
    g = s1["generated_solarwm"]
    check("P1 gen A median", g["A_flatness"]["median_clip_score"], 14.208, 1e-2)
    check("P1 gen B median", g["B_level"]["median_clip_score"], 25.936, 1e-2)
    check("P1 gen C median", g["mirrored"]["median_clip_score"], 22.198, 1e-2)
    check("P1 real calibrated tau", s1["real_calibrated"]["tau_window_deg_s"], 30.081, 1e-2)
    df = pd.read_csv(P1 / "per_clip.csv")
    check("P1 literal median uzh", df[df.kind == "uzh"].literal_abs.median(), 26.57, 0.01)
    f = df[(df.kind == "sim") & (df.label == "feasible")]
    check("P1 feasible median mount 0", f[f.mount == 0.0]["abs_w0.5"].median(), 0.97, 0.01)
    check("P1 feasible median mount 30", f[f.mount == 30.0]["abs_w0.5"].median(), 10.15, 0.01)
    check("P1 feasible median mount 45", f[f.mount == 45.0]["abs_w0.5"].median(), 16.50, 0.01)

    s2 = json.load(open(P2 / "summary.json"))
    check("P2 path defects detected", s2["path_defects"]["detected_by_physics_gate"], "13/13")
    check("P2 feasible flagged", s2["path_defects"]["feasible_flagged"], "0/80")
    check("P2 manifest diff", s2["path_defects"]["manifest_vs_recomputed_peak_tilt_max_abs_diff"],
          0.1820, 1e-3)
    check("P2 physics rejected", s2["physics_gate_real"]["rejected"], "11/20")
    check("P2 tilt rejected", s2["physics_gate_real"]["tilt_rejected"], "8/20")
    check("P2 twr rejected", s2["physics_gate_real"]["twr_rejected"], "9/20")
    check("P2 data tilt envelope", s2["real_envelope_max"]["tilt"], 122.53, 0.01)
    check("P2 data twr envelope", s2["real_envelope_max"]["twr"], 6.05, 0.01)
    check("P2 sim outside data envelope", s2["outside_envelope_sim_dev"]["tilt"], "0/200")
    check("P2 sim outside physics", s2["outside_envelope_sim_dev"]["physics_tilt"], "13/200")
    check("P2 gen outside", s2["outside_envelope_gen_h04"]["tilt"], "0/22")
    check("P2 conjunction", s2["conjunction_real"]["pass"], "9/20")
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print("\n".join(FAILS))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
