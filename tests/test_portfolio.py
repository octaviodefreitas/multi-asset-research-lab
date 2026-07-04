import numpy as np
import pandas as pd
import pytest

from engine import portfolio


@pytest.fixture
def returns():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=800)
    return pd.DataFrame({
        "CALM": rng.normal(0.0004, 0.005, 800),   # low vol, positive drift
        "WILD": rng.normal(0.0000, 0.020, 800),   # high vol, no drift
    }, index=idx)


def test_equal_weights_sum_to_one(returns):
    w = portfolio.equal_weights(returns)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_inverse_vol_favors_low_vol_asset(returns):
    w = portfolio.inverse_vol_weights(returns, lookback=63).dropna()
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert (w["CALM"] > w["WILD"]).all()
    # vol ratio ~4x should give weight ratio ~4x
    assert w["CALM"].mean() / w["WILD"].mean() == pytest.approx(4.0, rel=0.4)


def test_inverse_vol_no_lookahead(returns):
    w = portfolio.inverse_vol_weights(returns, lookback=63)
    tampered = returns.copy()
    tampered.iloc[500:] *= 10.0
    w2 = portfolio.inverse_vol_weights(tampered, lookback=63)
    pd.testing.assert_frame_equal(w.iloc[:500], w2.iloc[:500])


def test_tangency_gross_leverage_one_and_prefers_drift(returns):
    w = portfolio.tangency_weights(returns, lookback=252).dropna()
    assert len(w) > 0
    np.testing.assert_allclose(w.abs().sum(axis=1), 1.0)
    # the asset with positive drift and lower vol should dominate
    assert (w["CALM"].abs() > w["WILD"].abs()).mean() > 0.9


def test_tangency_weights_applied_after_estimation(returns):
    """Weights on any day must be estimable from strictly earlier data."""
    w = portfolio.tangency_weights(returns, lookback=252)
    tampered = returns.copy()
    tampered.iloc[600:] *= -5.0
    w2 = portfolio.tangency_weights(tampered, lookback=252)
    pd.testing.assert_frame_equal(w.iloc[:600], w2.iloc[:600])


def test_apply_weights_matches_manual(returns):
    w = portfolio.equal_weights(returns)
    port = portfolio.apply_weights(returns, w)
    expected = returns.mean(axis=1)
    np.testing.assert_allclose(port, expected)
