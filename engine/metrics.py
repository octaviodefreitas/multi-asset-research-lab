"""Performance metrics on simple daily returns. 252 periods/year, rf = 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    total = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualized_vol(returns)
    if np.isnan(vol) or vol == 0:
        return np.nan
    return returns.mean() * TRADING_DAYS / vol


def sortino_ratio(returns: pd.Series) -> float:
    """Downside deviation uses squared negative returns over ALL observations
    (the standard formulation), not the std of the loss subsample."""
    if len(returns) < 2:
        return np.nan
    downside = np.minimum(returns, 0.0)
    dvol = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if dvol == 0:
        return np.nan
    return returns.mean() * TRADING_DAYS / dvol


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    return drawdown_series(returns).min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if np.isnan(mdd) or mdd == 0:
        return np.nan
    return cagr(returns) / mdd


def hit_rate(returns: pd.Series, positions: pd.Series | None = None) -> float:
    """Fraction of positive-return days, counted only while a position is on."""
    if positions is not None:
        returns = returns[positions.reindex(returns.index).fillna(0.0) != 0]
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


def annualized_turnover(turnover: pd.Series) -> float:
    """Average one-sided daily turnover (|Δposition| per unit NAV), annualized."""
    if len(turnover) == 0:
        return np.nan
    return turnover.mean() * TRADING_DAYS


def benchmark_relative(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """CAPM-style statistics of a return stream measured against a benchmark.

    Beta: sensitivity to benchmark moves. Alpha: annualized return left over
    after removing beta × benchmark (the part not explained by market exposure).
    Tracking error: volatility of the active return. Information ratio: active
    return per unit of tracking error.
    """
    keys = ("Beta", "Alpha (ann.)", "Tracking Error", "Information Ratio", "Correlation")
    df = pd.concat([returns, benchmark], axis=1, keys=["r", "b"]).dropna()
    if len(df) < 2:
        return dict.fromkeys(keys, np.nan)
    r, b = df["r"], df["b"]
    var_b = b.var(ddof=1)
    beta = r.cov(b) / var_b if var_b > 0 else np.nan
    alpha = (r.mean() - beta * b.mean()) * TRADING_DAYS if not np.isnan(beta) else np.nan
    active = r - b
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ir = active.mean() * TRADING_DAYS / te if te > 0 else np.nan
    return {"Beta": beta, "Alpha (ann.)": alpha, "Tracking Error": te,
            "Information Ratio": ir, "Correlation": r.corr(b)}


def summary(returns: pd.Series, positions: pd.Series | None = None,
            turnover: pd.Series | None = None) -> dict[str, float]:
    return {
        "CAGR": cagr(returns),
        "Ann. Vol": annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Ann. Turnover": annualized_turnover(turnover) if turnover is not None else np.nan,
        "Hit Rate": hit_rate(returns, positions),
    }
