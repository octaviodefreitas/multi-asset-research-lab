import numpy as np
import pandas as pd
import pytest

from engine import signals


@pytest.fixture
def close():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=500)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(500, 2)), axis=0))
    return pd.DataFrame(prices, index=idx, columns=["A", "B"])


def test_ma_crossover_no_lookahead(close):
    """Changing the future must not change past signal values."""
    sig = signals.ma_crossover(close, 5, 20)
    tampered = close.copy()
    tampered.iloc[300:] *= 3.0
    sig_tampered = signals.ma_crossover(tampered, 5, 20)
    pd.testing.assert_frame_equal(sig.iloc[:300], sig_tampered.iloc[:300])


def test_momentum_no_lookahead(close):
    sig = signals.momentum(close, lookback=63)
    tampered = close.copy()
    tampered.iloc[300:] *= 0.1
    sig_tampered = signals.momentum(tampered, lookback=63)
    pd.testing.assert_frame_equal(sig.iloc[:300], sig_tampered.iloc[:300])


def test_ma_crossover_uptrend_is_long(close):
    trend = pd.DataFrame({"A": np.linspace(100, 200, 300)},
                         index=pd.bdate_range("2015-01-01", periods=300))
    sig = signals.ma_crossover(trend, 10, 50)
    assert (sig["A"].iloc[60:] == 1.0).all()


def test_momentum_downtrend_is_short():
    trend = pd.DataFrame({"A": np.linspace(200, 100, 300)},
                         index=pd.bdate_range("2015-01-01", periods=300))
    sig = signals.momentum(trend, lookback=63)
    assert (sig["A"].iloc[70:] == -1.0).all()


def test_long_only_never_short(close):
    sig = signals.ma_crossover(close, 10, 50, long_only=True)
    assert (sig.fillna(0) >= 0).all().all()
    sig = signals.momentum(close, 63, long_only=True)
    assert (sig.fillna(0) >= 0).all().all()


def test_invalid_windows_raise(close):
    with pytest.raises(ValueError):
        signals.ma_crossover(close, 50, 50)
    with pytest.raises(ValueError):
        signals.momentum(close, lookback=10, skip=10)


def test_mean_reversion_no_lookahead(close):
    sig = signals.mean_reversion(close, lookback=20, z_entry=1.0)
    tampered = close.copy()
    tampered.iloc[300:] *= 5.0
    sig_tampered = signals.mean_reversion(tampered, lookback=20, z_entry=1.0)
    pd.testing.assert_frame_equal(sig.iloc[:300], sig_tampered.iloc[:300])


def test_mean_reversion_fades_stretched_prices():
    idx = pd.bdate_range("2020-01-01", periods=40)
    flat = np.full(40, 100.0) + np.tile([0.1, -0.1], 20)  # tiny noise, well inside the band
    spike_up = flat.copy()
    spike_up[-1] = 120.0
    spike_down = flat.copy()
    spike_down[-1] = 80.0
    up = signals.mean_reversion(pd.DataFrame({"A": spike_up}, index=idx), 20, 1.0)
    down = signals.mean_reversion(pd.DataFrame({"A": spike_down}, index=idx), 20, 1.0)
    assert up["A"].iloc[-1] == -1.0    # stretched above the mean -> short
    assert down["A"].iloc[-1] == 1.0   # stretched below the mean -> long
    assert up["A"].iloc[-2] == 0.0     # inside the band -> flat


def test_mean_reversion_long_only_never_short(close):
    sig = signals.mean_reversion(close, 20, 1.0, long_only=True)
    assert (sig.fillna(0) >= 0).all().all()


def test_ichimoku_no_lookahead(close):
    sig = signals.ichimoku(close, conversion=9, base=26)
    tampered = close.copy()
    tampered.iloc[300:] *= 4.0
    sig_tampered = signals.ichimoku(tampered, conversion=9, base=26)
    pd.testing.assert_frame_equal(sig.iloc[:300], sig_tampered.iloc[:300])


def test_ichimoku_uptrend_above_cloud_is_long():
    trend = pd.DataFrame({"A": np.linspace(100, 300, 300)},
                         index=pd.bdate_range("2015-01-01", periods=300))
    sig = signals.ichimoku(trend, conversion=9, base=26)
    # warmup = span_b (52) + displacement (26); afterwards price sits above the cloud
    assert (sig["A"].iloc[100:] == 1.0).all()


def test_ichimoku_downtrend_below_cloud_is_short():
    trend = pd.DataFrame({"A": np.linspace(300, 100, 300)},
                         index=pd.bdate_range("2015-01-01", periods=300))
    sig = signals.ichimoku(trend, conversion=9, base=26)
    assert (sig["A"].iloc[100:] == -1.0).all()


def test_ichimoku_long_only_never_short(close):
    sig = signals.ichimoku(close, 9, 26, long_only=True)
    assert (sig.fillna(0) >= 0).all().all()


def test_vol_target_caps_leverage(close):
    rets = close.pct_change()
    sig = signals.ma_crossover(close, 5, 20)
    scaled = signals.vol_target(sig, rets, target_vol=0.50, lookback=20, max_leverage=2.0)
    assert scaled.abs().max().max() <= 2.0 + 1e-12
