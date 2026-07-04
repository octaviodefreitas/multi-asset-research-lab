"""Monte Carlo risk analysis via block bootstrap of historical returns."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bootstrap_paths(returns: pd.Series, horizon: int = 252, n_paths: int = 1000,
                    block: int = 21, seed: int = 42) -> np.ndarray:
    """Simulate forward equity paths by resampling contiguous blocks of
    historical daily returns (block bootstrap preserves short-range
    autocorrelation and volatility clustering that iid resampling destroys).

    Returns an array of shape (n_paths, horizon): growth of $1 per path.
    """
    r = returns.dropna().to_numpy()
    if len(r) <= block:
        raise ValueError("Not enough history to bootstrap from")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(horizon / block)
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
    sampled = r[idx[:, :horizon]]
    return np.cumprod(1.0 + sampled, axis=1)


def fan_percentiles(paths: np.ndarray, pcts=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Per-day percentiles across paths, for fan charts."""
    data = np.percentile(paths, pcts, axis=0)
    return pd.DataFrame(data.T, columns=[f"p{p}" for p in pcts],
                        index=np.arange(1, paths.shape[1] + 1))
