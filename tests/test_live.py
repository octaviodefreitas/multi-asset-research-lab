import numpy as np
import pandas as pd
import pytest

from engine import live


@pytest.fixture
def panel():
    rng = np.random.default_rng(10)
    idx = pd.bdate_range("2018-01-01", periods=1200)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(1200, 2)), axis=0))
    return pd.DataFrame(prices, index=idx, columns=["A", "B"])


@pytest.fixture
def config(panel):
    freeze = panel.index[1000]
    return {
        "freeze_date": str(freeze.date()),
        "signal_type": "MA Crossover",
        "params": {"short": 20, "long": 100},
        "direction": "long_short",
        "vol_target": False,
        "cost_bps": 5.0,
    }


def test_live_slice_strictly_after_freeze(panel, config):
    track = live.live_track(panel, config)
    assert (track["live"].index > track["freeze"]).all()
    assert len(track["live"]) == len(panel) - 1001


def test_live_returns_match_full_series(panel, config):
    """The live slice must be exactly the tail of the full portfolio series —
    reconstruction, not recomputation with different settings."""
    track = live.live_track(panel, config)
    tail = track["portfolio"].loc[track["portfolio"].index > track["freeze"]]
    pd.testing.assert_series_equal(track["live"], tail)


def test_current_positions_cover_universe(panel, config):
    track = live.live_track(panel, config)
    assert list(track["current_positions"].index) == list(panel.columns)


def test_real_config_is_valid():
    config = live.load_config()
    for key in ("freeze_date", "universe", "signal_type", "params",
                "direction", "cost_bps", "history_start"):
        assert key in config
    assert pd.Timestamp(config["freeze_date"]) <= pd.Timestamp.now()
