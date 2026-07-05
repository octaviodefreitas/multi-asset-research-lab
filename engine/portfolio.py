"""Portfolio construction: turning per-asset strategy returns into portfolio
weights. All schemes are causal, weights applied on day t are estimated from
data through day t-1 or earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """1/N over the assets that have started trading (non-NaN return streams)."""
    active = returns.notna()
    w = active.div(active.sum(axis=1), axis=0)
    return w.where(active)


def inverse_vol_weights(returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Risk-parity-style weights: each asset weighted by 1 / trailing vol, so
    every sleeve contributes similar risk. Shifted one day (no lookahead)."""
    vol = returns.rolling(lookback).std(ddof=1)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.shift(1)


def tangency_weights(returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Unconstrained mean-variance (tangency) weights w ∝ Σ⁻¹μ, re-estimated on
    the last day of each month from a trailing window and applied to the
    following month only. Normalized to gross leverage 1 (Σ|w| = 1).

    Deliberately naive, sample means are terrible forecasts, which is exactly
    the point of comparing this against 1/N.
    """
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    ends = list(returns.groupby(pd.Grouper(freq="ME")).tail(1).index)
    for i, me in enumerate(ends):
        window = returns.loc[:me].tail(lookback)
        if len(window) < lookback:
            continue
        cols = [c for c in window.columns
                if window[c].notna().all() and window[c].std(ddof=1) > 0]
        if not cols:
            continue
        mu = window[cols].mean().values
        cov = window[cols].cov().values
        w = np.linalg.pinv(cov) @ mu
        gross = np.abs(w).sum()
        if gross <= 1e-12:
            continue
        w = w / gross
        until = ends[i + 1] if i + 1 < len(ends) else returns.index[-1]
        mask = (returns.index > me) & (returns.index <= until)
        if mask.any():
            weights.loc[mask, cols] = np.tile(w, (int(mask.sum()), 1))
    return weights


def apply_weights(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio return under a weight panel; days without weights earn 0."""
    return (weights * returns).sum(axis=1).fillna(0.0)
