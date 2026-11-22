import numpy as np

from src.verdict.gate_a import (SignalStats, gate_a, implied_snr, signal_stats,
                                window_signal, windows)
from src.verdict.selective import (auroc, cluster_auroc, cluster_interval,
                                   conformal_quantile, crc_threshold, rate_interval,
                                   risk_coverage)


def test_auroc_perfect_and_tied():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-12
    assert np.isnan(auroc([0.5, 0.5], [0, 0]))


def test_auroc_matches_pairwise_count():
    rng = np.random.default_rng(0)
    s = rng.normal(size=50)
    y = (s + rng.normal(scale=0.8, size=50) > 0).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert abs(auroc(s, y) - brute) < 1e-12


def test_conformal_quantile_is_conservative_at_small_n():
    v = np.arange(20, dtype=float)
    q = conformal_quantile(v, alpha=0.05)
    assert q == 19.0          # ceil(21*0.95)=20 -> the 20th smallest of 0..19
    assert np.isnan(conformal_quantile(v[:5], 0.05))


def test_risk_coverage_counts_and_nan():
    score = np.array([1.0, 2.0, 3.0, np.nan])
    risk = np.array([1.0, 0.0, 0.0, 1.0])
    out = risk_coverage(score, risk, [0.5, 1.5, 3.5])
    assert out["n_total"] == 3
    assert out["n_accept"].tolist() == [3, 2, 0]
    assert out["coverage"].tolist() == [1.0, 2 / 3, 0.0]
    assert out["risk"][0] == 1 / 3 and out["risk"][1] == 0.0


def test_crc_threshold_reaches_and_does_not():
    score = np.arange(100, dtype=float)
    risk = (score < 5).astype(float)
    res = crc_threshold(score, risk, alpha=0.05, min_accept=10)
    assert res.reached
    # accepting 99 clips with 4 of the 5 failures still bounds the risk at (4+1)/(99+1)
    assert res.lam == 1.0 and res.bound <= 0.05
    all_in = crc_threshold(score, risk, alpha=0.001, min_accept=10)
    assert not all_in.reached
    bad = crc_threshold(score, np.ones(100), alpha=0.05, min_accept=10)
    assert not bad.reached and np.isnan(bad.lam)
    assert "no threshold" in bad.reason


def test_cluster_interval_resamples_flights():
    vals = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
    flights = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    out = cluster_interval(vals, flights, draws=200)
    assert out["point"] == vals.mean()
    assert out["n_clusters"] == 3
    assert "descriptive" in out["note"]
    rate = rate_interval(vals, flights, draws=200)
    assert abs(rate["point"] - vals.mean()) < 1e-12


def test_cluster_auroc_interval_contains_point():
    rng = np.random.default_rng(3)
    s = rng.normal(size=60)
    y = (s > 0).astype(int)
    fl = np.repeat([f"f{i}" for i in range(6)], 10)
    out = cluster_auroc(s, y, fl, draws=300)
    assert 0.5 < out["point"] <= 1.0
    assert out["lo"] <= out["point"] <= out["hi"]


def test_windows_match_published_scheme():
    assert windows(8) == [(0, 8)]
    assert windows(10) == [(0, 8), (2, 10)]
    w = windows(48)
    assert w[0] == (0, 8) and w[-1] == (40, 48)
    assert all(b - a == 8 for a, b in w)


def test_signal_stats_recovers_known_sigma():
    rng = np.random.default_rng(1)
    n = 4000
    sig = 0.10                      # 5.7 deg in lean units
    lean = np.column_stack([sig * rng.normal(size=n), 0.001 * rng.normal(size=n)])
    st = signal_stats(lean)
    assert abs(st.sigma_x - np.degrees(sig)) < 0.5
    assert abs(st.sigma_min - np.degrees(np.std(lean[:, 1]))) < 1e-12
    assert st.sigma_p > st.sigma_min


def test_signal_stats_flat_clip_is_weak_and_gate_abstains():
    lean = np.zeros((48, 2))
    tilt = np.zeros(48)
    st = signal_stats(lean, tilt)
    assert st.sigma_min < 1e-9
    g = gate_a(st, sigma_threshold=3.0)
    assert not g["testable"] and "weak_signal" in g["reasons"]


def test_near_vertical_is_a_reason():
    rng = np.random.default_rng(2)
    tilt = np.full(48, 85.0)
    lean = np.degrees(np.column_stack([0.2 * rng.normal(size=48), 0.2 * rng.normal(size=48)]))
    st = signal_stats(lean, tilt)
    g = gate_a(st, sigma_threshold=0.1)
    assert "near_vertical" in g["reasons"]
    assert st.frac_above_80 == 1.0


def test_implied_snr_limits():
    assert abs(implied_snr(3.0, 3.0) - 1 / np.sqrt(2)) < 1e-12
    assert implied_snr(0.0, 3.0) == 0.0
    assert implied_snr(1e6, 3.0) > 0.999
    assert np.isnan(implied_snr(np.nan, 3.0))


def test_window_signal_returns_one_row_per_window():
    rng = np.random.default_rng(4)
    lean = np.column_stack([0.1 * rng.normal(size=48), 0.1 * rng.normal(size=48)])
    rows = window_signal(lean)
    assert len(rows) == len(windows(48))
    assert {"sigma_min", "start", "stop"} <= set(rows[0])
    assert rows[0]["start"] == 0 and rows[-1]["stop"] == 48


def test_signal_stats_dataclass_serialises():
    st = SignalStats(1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 10)
    assert st.as_dict()["sigma_min"] == 1.0
