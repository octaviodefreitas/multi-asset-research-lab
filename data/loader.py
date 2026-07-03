"""Market data layer: daily OHLCV via yfinance with a local parquet cache.

The cache avoids re-hitting the Yahoo API on every Streamlit rerun; a stale
cache is still used as a fallback if the download fails (e.g. offline).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"

DEFAULT_START = "2006-01-01"

# Ticker -> human-readable label. Six assets, six distinct asset classes.
UNIVERSE = {
    "SPY": "US Equities (SPY)",
    "AGG": "US Bonds (AGG)",
    "GLD": "Gold (GLD)",
    "EURUSD=X": "EUR/USD FX Spot",
    "USO": "Crude Oil (USO)",
    "BTC-USD": "Bitcoin (BTC-USD)",
}


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _download(ticker: str, start: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    df = raw[cols].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def load_prices(ticker: str, start: str = DEFAULT_START, max_age_hours: float = 24.0) -> pd.DataFrame:
    """Daily OHLCV for one ticker, served from the parquet cache when fresh."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(ticker)
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            return pd.read_parquet(path)
    try:
        df = _download(ticker, start)
        df.to_parquet(path)
        return df
    except Exception:
        if path.exists():
            return pd.read_parquet(path)
        raise


def load_universe(tickers: list[str], start: str = DEFAULT_START) -> pd.DataFrame:
    """Aligned close-price panel (columns = tickers).

    Aligned on weekdays; assets with different calendars (FX, crypto) are
    forward-filled up to 5 days so cross-asset dates line up. Leading NaNs are
    kept so each asset only enters the backtest after its own inception.
    """
    closes = {t: load_prices(t, start)["Close"] for t in tickers}
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[panel.index.dayofweek < 5]
    return panel.ffill(limit=5)
