"""Vectorized systematic signals.

All signals are causal: the value at date t uses information up to and
including the close at t. The backtest shifts positions by one bar, so a
signal observed at t earns the return from t to t+1 — no lookahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover(close: pd.DataFrame, short: int, long: int, long_only: bool = False) -> pd.DataFrame:
    """Moving-average crossover: +1 when the short SMA is above the long SMA,
    -1 below (0 if long_only). short=50 / long=200 is the classic golden cross."""
    if short >= long:
        raise ValueError(f"short window ({short}) must be < long window ({long})")
    fast = close.rolling(short).mean()
    slow = close.rolling(long).mean()
    sig = np.sign(fast - slow)
    if long_only:
        sig = sig.clip(lower=0.0)
    return sig


def momentum(close: pd.DataFrame, lookback: int = 126, skip: int = 0, long_only: bool = False) -> pd.DataFrame:
    """Time-series momentum: sign of the trailing return over `lookback` bars,
    optionally skipping the most recent `skip` bars (short-term reversal filter)."""
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be < lookback ({lookback})")
    mom = close.shift(skip) / close.shift(lookback) - 1.0
    sig = np.sign(mom)
    if long_only:
        sig = sig.clip(lower=0.0)
    return sig


def mean_reversion(close: pd.DataFrame, lookback: int = 20, z_entry: float = 1.0,
                   long_only: bool = False) -> pd.DataFrame:
    """Z-score mean reversion: fade stretched prices. When price sits more than
    `z_entry` standard deviations above its rolling mean, go short (it is
    expected to fall back); more than `z_entry` below, go long; flat inside the
    band. The countertrend counterpart to the two trend signals."""
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std(ddof=1)
    z = (close - mean) / std
    sig = (-np.sign(z)).where(z.abs() > z_entry, 0.0)
    if long_only:
        sig = sig.clip(lower=0.0)
    return sig


def _midpoint(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return (close.rolling(window).max() + close.rolling(window).min()) / 2.0


def ichimoku(close: pd.DataFrame, conversion: int = 9, base: int = 26,
             span_b: int | None = None, long_only: bool = False) -> pd.DataFrame:
    """Ichimoku Kinko Hyo cloud signal (Goichi Hosoda's classic 9/26/52 system).

    Long when price trades above the cloud, short below it, flat inside it.
    The cloud spans are computed from data available `base` days ago (the
    classic forward displacement), so the signal at t is strictly causal.
    Midpoints use rolling closes (the textbook version uses high/low; on a
    close-only panel this is the standard approximation).
    """
    if span_b is None:
        span_b = 2 * base
    tenkan = _midpoint(close, conversion)
    kijun = _midpoint(close, base)
    span_a = ((tenkan + kijun) / 2.0).shift(base)
    span_b_line = _midpoint(close, span_b).shift(base)
    upper = np.maximum(span_a, span_b_line)
    lower = np.minimum(span_a, span_b_line)
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig = sig.mask(close > upper, 1.0).mask(close < lower, -1.0)
    sig = sig.where(upper.notna())
    if long_only:
        sig = sig.clip(lower=0.0)
    return sig


def combined(close: pd.DataFrame, short: int, long: int, lookback: int,
             long_only: bool = False) -> pd.DataFrame:
    """Equal-weight blend of MA crossover and momentum. Positions are
    fractional when the two signals disagree (conviction weighting)."""
    return 0.5 * (ma_crossover(close, short, long, long_only)
                  + momentum(close, lookback, long_only=long_only))


def vol_target(signal: pd.DataFrame, returns: pd.DataFrame, target_vol: float = 0.10,
               lookback: int = 20, max_leverage: float = 2.0) -> pd.DataFrame:
    """Volatility-targeting overlay: scale positions by target / realized vol,
    capped at max_leverage. Realized vol at t only uses returns up to t."""
    realized = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
    scale = (target_vol / realized).clip(upper=max_leverage)
    return signal * scale


SIGNAL_TYPES = ("MA Crossover", "Time-Series Momentum", "Mean Reversion (Z-Score)",
                "Ichimoku Cloud", "Combined")


def build_signal(close: pd.DataFrame, signal_type: str, long_only: bool = False,
                 vol_target_on: bool = False, target_vol: float = 0.10,
                 vt_lookback: int = 20, **params) -> pd.DataFrame:
    """Assemble the full position signal from a signal type + parameters.
    Used both by the app (slider values) and the walk-forward optimizer (grids)."""
    if signal_type == "MA Crossover":
        sig = ma_crossover(close, params["short"], params["long"], long_only)
    elif signal_type == "Time-Series Momentum":
        sig = momentum(close, params["lookback"], params.get("skip", 0), long_only)
    elif signal_type == "Mean Reversion (Z-Score)":
        sig = mean_reversion(close, params["lookback"], params["z_entry"], long_only)
    elif signal_type == "Ichimoku Cloud":
        sig = ichimoku(close, params["conversion"], params["base"], long_only=long_only)
    elif signal_type == "Combined":
        sig = combined(close, params["short"], params["long"], params["lookback"], long_only)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    if vol_target_on:
        rets = close.pct_change(fill_method=None)
        sig = vol_target(sig, rets, target_vol, vt_lookback)
    return sig
