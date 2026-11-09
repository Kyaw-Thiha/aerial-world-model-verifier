"""Tests for the label-free confidence gate (amendment 2).

The gate decides testability from pixels and estimator outputs only; these tests fix that
property, the label arithmetic, and the threshold rule. They use synthetic arrays, so they
run in the CPU environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from src.perception.frames import camera_to_body_pitched
from src.perception.gate import (MIN_KEEP, SIGNALS, SINGLE_FRAME, auroc_pred,
                                bootstrap_auroc_ci, clip_label, clip_signal, kept,
                                lofo_tau, pick_tau, s_temporal_jitter,
                                s_uncertainty_instability, s_vo_agreement,
                                up_truth_in_camera)

ROOT = Path(__file__).resolve().parents[1]


def level_quat(n: int = 1) -> np.ndarray:
    return np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))


def test_level_camera_up_is_minus_camera_y():
    up = up_truth_in_camera(level_quat(), uptilt_deg=0.0)
    np.testing.assert_allclose(up[0], [0.0, -1.0, 0.0], atol=1e-12)


def test_mount_pitch_round_trip():
    """A level body seen through a 50 deg mount puts world up along the mount's body plane."""
    up = up_truth_in_camera(level_quat(), uptilt_deg=50.0)
    R_cb = camera_to_body_pitched(np.deg2rad(50.0))
    np.testing.assert_allclose(up[0], R_cb.T @ np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.isclose(np.linalg.norm(up[0]), 1.0)


def test_truth_and_estimate_agree_means_zero_error():
    R = Rotation.from_euler("XZ", [12.0, -8.0], degrees=True).as_matrix()
    quat = Rotation.from_matrix(R).as_quat()
    truth = up_truth_in_camera(np.tile(quat, (5, 1)), uptilt_deg=30.0)
    out = clip_label(truth.copy(), truth, threshold_deg=10.0)
    assert out["median_error_deg"] < 1e-9
    assert out["label"] == 0
    assert out["n_frames"] == 5


def test_label_flips_at_the_threshold():
    truth = np.tile([0.0, 0.0, 1.0], (4, 1))
    off_by_11 = Rotation.from_euler("X", 11.0, degrees=True).as_matrix() @ truth.T
    out = clip_label(off_by_11.T, truth, threshold_deg=10.0)
    assert out["label"] == 1
    assert 10.9 < out["median_error_deg"] < 11.1


def test_label_is_none_when_nothing_is_finite():
    out = clip_label(np.full((3, 3), np.nan), np.tile([0.0, 0.0, 1.0], (3, 1)))
    assert out["label"] is None and out["n_frames"] == 0


def test_jitter_is_zero_for_a_constant_estimate():
    up = np.tile([0.0, -1.0, 0.0], (12, 1))
    assert s_temporal_jitter(up, dt_s=0.0833) == pytest.approx(0.0, abs=1e-9)


def test_jitter_grows_with_frame_to_frame_noise():
    rng = np.random.default_rng(0)
    base = np.tile([0.0, -1.0, 0.0], (60, 1))
    noisy = base + rng.normal(scale=0.02, size=base.shape)
    assert s_temporal_jitter(noisy, dt_s=0.0833) > s_temporal_jitter(base, dt_s=0.0833)
    assert s_temporal_jitter(noisy, dt_s=0.0833) > 0.1


def test_jitter_window_is_defined_in_seconds():
    """Same wobble, coarser sampling: fewer samples in the 0.5 s window, more jitter."""
    up = np.tile([0.0, -1.0, 0.0], (24, 1))
    up[::2] = (Rotation.from_euler("X", 3.0, degrees=True).as_matrix() @ up[::2].T).T
    fine = s_temporal_jitter(up, dt_s=0.0833)       # 0.5 s -> 6 samples
    mid = s_temporal_jitter(up, dt_s=0.25)          # 0.5 s -> 2 samples: undefined
    coarse = s_temporal_jitter(up, dt_s=1.0)        # 0.5 s -> 1 sample: undefined
    assert np.isfinite(fine) and fine > 0.1
    assert np.isnan(mid), "a 2-sample window leaves nothing to smooth against"
    assert np.isnan(coarse), "a 1 s step cannot support a 0.5 s window"


def test_vo_agreement_zero_when_vo_explains_the_motion():
    up = np.tile([0.0, -1.0, 0.0], (5, 1))
    R_rel = np.tile(np.eye(3), (4, 1, 1))
    assert s_vo_agreement(up, R_rel) == pytest.approx(0.0, abs=1e-9)


def test_vo_agreement_reads_a_rotation_the_estimate_missed():
    up = np.tile([0.0, -1.0, 0.0], (2, 1))
    R_rel = Rotation.from_euler("X", 5.0, degrees=True).as_matrix()[None]
    assert s_vo_agreement(up, R_rel) == pytest.approx(5.0, abs=1e-6)


def test_vo_agreement_is_not_testable_without_pairs():
    assert np.isnan(s_vo_agreement(np.tile([0.0, -1.0, 0.0], (1, 1)), np.zeros((0, 3, 3))))


def test_uncertainty_instability_counts_outliers():
    assert s_uncertainty_instability(np.ones(10)) == 0.0
    u = np.ones(10)
    u[0] = 3.0
    assert s_uncertainty_instability(u) == pytest.approx(0.1)


def test_tau_separates_when_it_can_and_respects_min_keep():
    values = np.array([1.0, 1.1, 1.2, 5.0, 6.0])
    labels = np.array([0, 0, 0, 1, 1])
    tau = pick_tau(values, labels, min_keep=0.6)
    assert tau >= 1.2 and tau < 5.0          # keeps the three good clips
    assert kept(1.2, tau) and not kept(5.0, tau)


def test_tau_never_keeps_fewer_than_min_keep():
    values = np.array([0.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 1, 1, 0])
    tau = pick_tau(values, labels, min_keep=MIN_KEEP)
    assert (values <= tau).mean() >= MIN_KEEP


def test_kept_is_false_for_a_missing_signal():
    assert not kept(float("nan"), 1.0)
    assert not kept(1.0, float("nan"))


def test_auroc_pred_is_one_for_perfect_separation():
    assert auroc_pred(np.array([5.0, 6.0, 1.0, 2.0]), np.array([1, 1, 0, 0])) == 1.0
    assert auroc_pred(np.array([1.0, 2.0, 5.0, 6.0]), np.array([1, 1, 0, 0])) == 0.0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    flights = np.repeat(np.array(["a", "b", "c", "d"]), 10)
    labels = (flights == "a").astype(float)
    values = labels * 3.0 + rng.normal(size=40)
    ci = bootstrap_auroc_ci(values, labels, flights, n_boot=200, seed=0)
    assert ci["lo"] <= ci["auroc"] <= ci["hi"]
    assert ci["auroc"] > 0.8


def test_lofo_tau_holds_each_flight_out():
    values = np.array([1.0, 2.0, 3.0, 9.0])
    labels = np.array([0, 0, 0, 1])
    flights = np.array(["a", "a", "b", "b"])
    out = lofo_tau(values, labels, flights)
    assert set(out) == {"a", "b"}
    assert out["b"]["n_fit"] == 2


def test_frozen_gate_config_is_well_formed_if_present():
    p = ROOT / "configs/gate_frozen.yaml"
    if not p.exists():
        pytest.skip("gate not frozen yet")
    doc = yaml.safe_load(p.read_text())
    assert doc["signal"] in SIGNALS
    assert SINGLE_FRAME[doc["signal"]] is True
    assert np.isfinite(float(doc["tau"]))
    assert doc["prereg_amendment"] == 2
    assert doc["commit"], "the freezing commit must be recorded"


def test_gate_module_cannot_reach_the_path_or_mocap():
    """Code, not discipline: the gate's module must not import datasets, physics or fusion."""
    src = (ROOT / "src/perception/gate.py").read_text()
    for forbidden in ("src.datasets", "src.physics", "src.fusion", "src.score"):
        assert f"import {forbidden}" not in src
        assert f"from {forbidden}" not in src
