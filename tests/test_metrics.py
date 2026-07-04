import numpy as np
import pandas as pd
import pytest

from engine import metrics


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_cagr_constant_return():
    r = series([0.001] * 252)  # exactly one trading year
    assert metrics.cagr(r) == pytest.approx(1.001 ** 252 - 1)


def test_vol_and_sharpe_known_values():
    r = series([0.01, -0.01] * 126)
    expected_vol = r.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_vol(r) == pytest.approx(expected_vol)
    assert metrics.sharpe_ratio(r) == pytest.approx(r.mean() * 252 / expected_vol)


def test_sharpe_nan_when_zero_vol():
    assert np.isnan(metrics.sharpe_ratio(series([0.0] * 100)))


def test_max_drawdown_known_path():
    # equity: 1.10 -> 0.55 (-50% from peak) -> recovery
    r = series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_drawdown_zero_for_monotonic_gains():
    r = series([0.01] * 50)
    assert metrics.max_drawdown(r) == 0.0


def test_calmar_consistency():
    r = series([0.10, -0.50, 0.20] * 30)
    assert metrics.calmar_ratio(r) == pytest.approx(
        metrics.cagr(r) / abs(metrics.max_drawdown(r)))


def test_sortino_nan_without_losses():
    assert np.isnan(metrics.sortino_ratio(series([0.01] * 100)))


def test_hit_rate_only_counts_active_days():
    r = series([0.01, -0.01, 0.02, 0.03])
    pos = series([1.0, 1.0, 0.0, 0.0])  # only the first two days are invested
    assert metrics.hit_rate(r, pos) == pytest.approx(0.5)


def test_benchmark_relative_known_regression():
    rng = np.random.default_rng(3)
    b = series(rng.normal(0.0004, 0.01, 500))
    r = 0.5 * b + 0.0002  # exact linear relation: beta 0.5, daily alpha 2 bps
    rel = metrics.benchmark_relative(r, b)
    assert rel["Beta"] == pytest.approx(0.5)
    assert rel["Alpha (ann.)"] == pytest.approx(0.0002 * 252)
    assert rel["Correlation"] == pytest.approx(1.0)


def test_benchmark_relative_identical_series():
    b = series(np.random.default_rng(4).normal(0, 0.01, 300))
    rel = metrics.benchmark_relative(b, b)
    assert rel["Beta"] == pytest.approx(1.0)
    assert rel["Alpha (ann.)"] == pytest.approx(0.0)
    assert rel["Tracking Error"] == pytest.approx(0.0)


def test_annualized_turnover():
    t = series([0.02] * 100)
    assert metrics.annualized_turnover(t) == pytest.approx(0.02 * 252)
