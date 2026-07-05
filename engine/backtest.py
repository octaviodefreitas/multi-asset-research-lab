"""Backtest engine: signals -> positions -> net returns, plus walk-forward validation.

Position convention: signal observed at the close of t is traded into at the
close of t and earns the return from t to t+1 (implemented as shift(1)).
Transaction costs are charged on turnover: |Δposition| × cost_bps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import metrics


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    asset_returns: pd.DataFrame
    strategy_returns: pd.DataFrame
    turnover: pd.DataFrame
    equity: pd.DataFrame


def run_backtest(close: pd.DataFrame, signal: pd.DataFrame, cost_bps: float = 5.0) -> BacktestResult:
    """Vectorized backtest of one signal panel over a close-price panel.

    Assets contribute zero position (not zero cost, there is nothing to trade)
    before their first available price.
    """
    signal = signal.reindex(close.index)
    asset_rets = close.pct_change(fill_method=None)

    positions = signal.shift(1)
    positions = positions.where(asset_rets.notna()).fillna(0.0)
    rets = asset_rets.fillna(0.0)

    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0].abs()

    strategy = positions * rets - turnover * cost_bps / 1e4
    equity = (1.0 + strategy).cumprod()
    return BacktestResult(positions, rets, strategy, turnover, equity)


def equal_weight_returns(strategy_returns: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio of per-asset strategies, averaging only over
    assets that have started trading (valid = close price exists)."""
    active = strategy_returns.where(valid.reindex(strategy_returns.index))
    return active.mean(axis=1).fillna(0.0)


@dataclass
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    train_sharpe: float
    test_sharpe: float


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    splits: list[WalkForwardSplit] = field(default_factory=list)


def walk_forward(close: pd.DataFrame, signal_builder, param_grid: list[dict],
                 n_splits: int = 4, cost_bps: float = 5.0) -> WalkForwardResult:
    """Expanding-window walk-forward validation.

    The sample is cut into n_splits + 1 contiguous segments. Split k trains on
    segments [0, k) and tests on segment k, so every parameter choice is made
    strictly before the data it is evaluated on. Rolling signals are causal,
    so computing them once over the full history introduces no leakage, only
    the parameter *selection* is restricted to the training window.
    """
    valid = close.notna()
    portfolios: dict[tuple, pd.Series] = {}
    for params in param_grid:
        sig = signal_builder(close, **params)
        bt = run_backtest(close, sig, cost_bps)
        portfolios[_key(params)] = equal_weight_returns(bt.strategy_returns, valid)

    bounds = np.linspace(0, len(close), n_splits + 2).astype(int)
    splits: list[WalkForwardSplit] = []
    oos_parts: list[pd.Series] = []

    for k in range(1, n_splits + 1):
        train = slice(0, bounds[k])
        test = slice(bounds[k], bounds[k + 1])

        best = max(param_grid, key=lambda p: _safe_sharpe(portfolios[_key(p)].iloc[train]))
        test_returns = portfolios[_key(best)].iloc[test]
        oos_parts.append(test_returns)

        splits.append(WalkForwardSplit(
            train_start=close.index[0],
            train_end=close.index[bounds[k] - 1],
            test_start=close.index[bounds[k]],
            test_end=close.index[bounds[k + 1] - 1],
            params=best,
            train_sharpe=_safe_sharpe(portfolios[_key(best)].iloc[train]),
            test_sharpe=_safe_sharpe(test_returns),
        ))

    return WalkForwardResult(oos_returns=pd.concat(oos_parts), splits=splits)


def _key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def _safe_sharpe(returns: pd.Series) -> float:
    s = metrics.sharpe_ratio(returns)
    return -np.inf if np.isnan(s) else s
