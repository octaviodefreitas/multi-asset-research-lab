import numpy as np
import pandas as pd
import pytest

from data import loader


@pytest.fixture
def fake_history():
    idx = pd.bdate_range("2006-01-02", "2024-12-31")
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": 1e6}, index=idx)


@pytest.fixture
def patched(monkeypatch, tmp_path, fake_history):
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_download(ticker):
        calls["n"] += 1
        return fake_history

    monkeypatch.setattr(loader, "_download", fake_download)
    return calls


def test_start_date_is_respected(patched):
    df = loader.load_prices("FAKE", start="2015-01-01")
    assert df.index[0] >= pd.Timestamp("2015-01-01")
    assert df.index[0] <= pd.Timestamp("2015-01-05")


def test_changing_start_reslices_from_cache_without_redownload(patched):
    """Regression test: a different start date must change the returned range,
    served by slicing the cached full history (no second download)."""
    early = loader.load_prices("FAKE", start="2008-01-01")
    late = loader.load_prices("FAKE", start="2018-01-01")
    assert patched["n"] == 1  # one download, second call served from cache
    assert early.index[0].year == 2008
    assert late.index[0].year == 2018
    assert len(early) > len(late)


def test_stale_cache_fallback_when_download_fails(patched, monkeypatch):
    loader.load_prices("FAKE", start="2010-01-01")  # populate cache

    def boom(ticker):
        raise ConnectionError("offline")

    monkeypatch.setattr(loader, "_download", boom)
    # force the cache to look stale so a re-download is attempted and fails
    df = loader.load_prices("FAKE", start="2012-01-01", max_age_hours=0.0)
    assert df.index[0].year == 2012  # stale cache still served, correctly sliced
