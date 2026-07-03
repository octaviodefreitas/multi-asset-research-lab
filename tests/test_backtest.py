import numpy as np
import pandas as pd
import pytest

from engine import backtest, signals


@pytest.fixture
def close():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012-01-01", periods=600)
    prices = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=(600, 2)), axis=0))
    return pd.DataFrame(prices, index=idx, columns=["A", "B"])


def test_always_long_no_cost_equals_buy_and_hold(close):
    """Constant +1 signal with zero costs must reproduce buy & hold."""
    sig = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    bt = backtest.run_backtest(close, sig, cost_bps=0.0)
    expected = close.iloc[-1] / close.iloc[0]
    pd.testing.assert_series_equal(bt.equity.iloc[-1], expected, check_names=False)


def test_costs_strictly_reduce_returns(close):
    sig = signals.ma_crossover(close, 5, 20)
    free = backtest.run_backtest(close, sig, cost_bps=0.0)
    costly = backtest.run_backtest(close, sig, cost_bps=10.0)
    assert costly.equity.iloc[-1].lt(free.equity.iloc[-1]).all()


def test_turnover_math():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104]}, index=idx)
    sig = pd.DataFrame({"A": [0.0, 1, 1, -1, -1]}, index=idx)
    bt = backtest.run_backtest(close, sig, cost_bps=0.0)
    # positions are signal shifted one day: [0, 0, 1, 1, -1]
    assert bt.positions["A"].tolist() == [0, 0, 1, 1, -1]
    # turnover: enter long (1), then flip long->short (2)
    assert bt.turnover["A"].tolist() == [0, 0, 1, 0, 2]


def test_position_lagged_one_bar():
    """The signal on day t must NOT earn day t's return (no lookahead)."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 110, 110, 110]}, index=idx)
    # Signal fires the same day as the +10% move...
    sig = pd.DataFrame({"A": [0.0, 1, 0, 0]}, index=idx)
    bt = backtest.run_backtest(close, sig, cost_bps=0.0)
    # ...so the strategy must capture none of it.
    assert bt.strategy_returns["A"].abs().sum() == 0.0


def test_no_position_before_inception(close):
    late = close.copy()
    late.iloc[:100, 1] = np.nan  # asset B starts 100 days later
    sig = pd.DataFrame(1.0, index=late.index, columns=late.columns)
    bt = backtest.run_backtest(late, sig, cost_bps=0.0)
    assert (bt.positions["B"].iloc[:100] == 0).all()
    assert (bt.strategy_returns["B"].iloc[:100] == 0).all()


def test_walk_forward_covers_sample_without_overlap(close):
    builder = lambda c, **p: signals.ma_crossover(c, p["short"], p["long"])
    grid = [{"short": 5, "long": 20}, {"short": 10, "long": 50}]
    wf = backtest.walk_forward(close, builder, grid, n_splits=3, cost_bps=5.0)
    assert not wf.oos_returns.index.duplicated().any()
    assert wf.oos_returns.index[-1] == close.index[-1]
    for a, b in zip(wf.splits[:-1], wf.splits[1:]):
        assert a.test_end < b.test_start
    for s in wf.splits:
        assert s.train_end < s.test_start  # params always chosen before the test window
