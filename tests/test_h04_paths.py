"""H04 path and scoring-schedule checks (CPU, no GPU, no weights)."""
from __future__ import annotations

import numpy as np

from experiments.h04_paths import (CLIP_S, DT, MAX_RATE_DEG_PER_FRAME, PATHS, check,
                                    f95_and_window, required_side)
from experiments.h04_score import parse_sid


def test_every_path_passes_the_design_checks():
    t = np.arange(0.0, CLIP_S, DT)
    for name, family, target in PATHS:
        info = check(name, family, target, t)
        assert info["peak_thrust_ratio"] < 2.0, name
        assert info["max_rel_translation_m"] <= 16.0, name
        assert max(info["max_roll_rate_deg_per_frame"],
                   info["max_pitch_rate_deg_per_frame"]) <= MAX_RATE_DEG_PER_FRAME, name
        achieved = (info["achieved_peak_pitch_deg"] if info["design_metric"] == "pitch"
                    else info["achieved_peak_bank_deg"])
        assert abs(achieved - target) <= 0.2, (name, achieved, target)


def test_required_side_is_unit_bounded_and_has_interior():
    t = np.arange(0.0, CLIP_S, DT)
    info = check("s_10", "s", 10.0, t)
    from experiments.h04_paths import positions
    pos = positions("s", info["solved_amplitude"], t)
    win = f95_and_window(pos, DT)
    assert np.isfinite(win["f95_hz"]) and win["clamped_s"] > 0.05
    rs = required_side(pos, win["clamped_s"], DT)
    lean = rs["lean_req"]
    finite = np.isfinite(lean).all(axis=1)
    assert finite.sum() > 60
    assert np.nanmax(np.linalg.norm(lean, axis=1)) <= 1.0 + 1e-9
    assert np.nanmax(rs["tilt_req_deg"]) > 5.0


def test_parse_sid():
    assert parse_sid("main_arc_5_A_flatness_g2") == ("arc_5", "A_flatness", 2, "")
    assert parse_sid("main_accel_brake_10_B_level_g0") == ("accel_brake_10", "B_level", 0, "")
    assert parse_sid("swap_slalom_10_F4_C_mirrored_roll_g0") == (
        "slalom_10", "C_mirrored_roll", 0, "F4")
    assert parse_sid("probe_p20") is None
