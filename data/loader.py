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

DEFAULT_START = "2000-01-01"

# Ticker -> human-readable label, grouped by asset class.
UNIVERSE = {
    "SPY": "US Equities (SPY)",
    "EFA": "Intl Developed Equities (EFA)",
    "EEM": "Emerging Markets (EEM)",
    "AGG": "US Aggregate Bonds (AGG)",
    "TLT": "Long-Term US Treasuries (TLT)",
    "HYG": "High-Yield Credit (HYG)",
    "GLD": "Gold (GLD)",
    "USO": "Crude Oil (USO)",
    "DBC": "Broad Commodities (DBC)",
    "VNQ": "US Real Estate (VNQ)",
    "EURUSD=X": "EUR/USD FX Spot",
    "GBPUSD=X": "GBP/USD FX Spot",
    "JPY=X": "USD/JPY FX Spot",
    "BTC-USD": "Bitcoin (BTC-USD)",
}


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_v3.parquet"  # v3: full history from 2000


def _download(ticker: str) -> pd.DataFrame:
    # Always fetch from DEFAULT_START so one cached file serves any requested
    # start date via slicing — the cache must never depend on the request.
    raw = yf.download(ticker, start=DEFAULT_START, auto_adjust=True, progress=False)
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
    """Daily OHLCV for one ticker from `start` onwards.

    The parquet cache always holds the full history (from DEFAULT_START), and
    the requested start date is applied by slicing — so changing `start` never
    requires a re-download and always takes effect.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(ticker)
    df = None
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            df = pd.read_parquet(path)
    if df is None:
        try:
            df = _download(ticker)
            df.to_parquet(path)
        except Exception:
            if path.exists():
                df = pd.read_parquet(path)
            else:
                raise
    return df.loc[df.index >= pd.Timestamp(start)]


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
