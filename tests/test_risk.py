import numpy as np
import pandas as pd
import pytest

from engine import risk


@pytest.fixture
def returns():
    rng = np.random.default_rng(6)
    return pd.Series(rng.normal(0.0004, 0.01, 1000),
                     index=pd.bdate_range("2015-01-01", periods=1000))


def test_paths_shape_and_positive(returns):
    paths = risk.bootstrap_paths(returns, horizon=252, n_paths=200, seed=1)
    assert paths.shape == (200, 252)
    assert (paths > 0).all()


def test_seed_reproducibility(returns):
    a = risk.bootstrap_paths(returns, horizon=100, n_paths=50, seed=7)
    b = risk.bootstrap_paths(returns, horizon=100, n_paths=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_median_final_value_tracks_drift(returns):
    paths = risk.bootstrap_paths(returns, horizon=252, n_paths=2000, seed=2)
    expected = (1 + returns.mean()) ** 252
    assert np.median(paths[:, -1]) == pytest.approx(expected, rel=0.10)


def test_fan_percentiles_ordered(returns):
    paths = risk.bootstrap_paths(returns, horizon=100, n_paths=500, seed=3)
    fan = risk.fan_percentiles(paths)
    assert (fan["p5"] <= fan["p50"]).all() and (fan["p50"] <= fan["p95"]).all()


def test_insufficient_history_raises():
    short = pd.Series([0.01] * 10)
    with pytest.raises(ValueError):
        risk.bootstrap_paths(short, block=21)
