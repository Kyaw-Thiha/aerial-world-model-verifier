"""End-to-end test of the Gate 3 evaluation on synthetic records with a known answer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p3a_verdict", ROOT / "experiments/phase3a_verdict.py")
verdict_exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict_exp)

PREREG = {
    "thresholds": {"speed_bins_mps": [[0, 2], [2, 5], [5, 10], [10, 20], [20, 99]],
                   "alpha": 0.05, "threshold_fit_flights": "all_flights"},
    "abstention": {},
}


def _records(n_flights=6, n_clips=12, gap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for f in range(n_flights):
        flight = f"flight-{f:02d}"
        track = "trackRATM" if f >= n_flights - 1 else "ellipse"
        for c in range(n_clips):
            base = rng.normal(4.0, 1.0)
            speed = float(rng.uniform(2, 9))
            for cond in ("identity", "tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll",
                         "yaw_control"):
                score = base if cond in ("identity", "yaw_control") else base + gap
                recs.append({"condition": cond, "flight": flight, "clip_id": f"{flight}#{c:02d}",
                             "track": track, "control": "autonomous", "speed": speed,
                             "bin": "2-5" if speed < 5 else "5-10",
                             "testable": True, "score": float(score),
                             "score_p75": float(score) + 1.0,
                             "score_median_tilt": float(score),
                             "tilt_implied_median": 20.0, "valid_frac_mean": 1.0,
                             "edit_angle_deg": 0.0 if cond == "identity" else 20.0,
                             "nuisances": {"ok": True, "loss": 0.0, "uptilt_offset_deg": 0.0,
                                           "lag_ms": 0.0}})
    return recs


def test_separation_shows_up_at_every_level():
    ev = verdict_exp.evaluation(_records(gap=25.0), PREREG)
    for cond in ("tilt_k0", "tilt_k05", "tilt_k15"):
        fam = ev["families"][cond]
        assert fam["auroc"] == pytest.approx(1.0)
        assert fam["rejection_rate"] == pytest.approx(1.0)
    # the control must be accepted at the real rate
    ctrl = ev["families"]["yaw_control"]
    assert ctrl["rejection_rate"] < 0.10
    assert ev["family_auroc_pooled"] == pytest.approx(1.0)
    assert ev["gate3"]["control_accepted"] is True


def test_no_separation_shows_up_as_chance():
    ev = verdict_exp.evaluation(_records(gap=0.0), PREREG)
    fam = ev["families"]["tilt_k0"]
    assert 0.35 < fam["auroc"] < 0.65
    assert ev["gate3"]["auroc_above_0.9_tilt_families"] is False


def test_thresholds_are_per_fold_and_finite_with_enough_clips():
    recs = _records(n_flights=6, n_clips=12)
    th = verdict_exp.thresholds_for(recs, PREREG, holdout=None)
    assert np.isfinite(th["pooled"]["threshold"])
    # leaving one flight out must not use that flight's own scores as the threshold
    th_f = verdict_exp.thresholds_for(recs, PREREG, holdout="flight-00")
    own = [r["score"] for r in recs if r["condition"] == "identity"
           and r["flight"] == "flight-00"]
    assert th["pooled"]["threshold"] >= th_f["pooled"]["threshold"] - 3 * np.std(own)


def test_trackratm_split_is_reported():
    ev = verdict_exp.evaluation(_records(gap=25.0), PREREG)
    assert ev["n_trackratm"] > 0 and ev["n_other"] > 0
    assert ev["heldout_acceptance_other"] is not None
    assert ev["heldout_acceptance_trackratm"] is not None
