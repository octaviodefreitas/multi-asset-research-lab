"""Fama-French factor data and factor regression.

Daily factor returns come from Ken French's data library (free, the academic
standard). The regression decomposes a strategy's returns into compensation
for known risk factors (market, size, value, momentum) plus alpha — the part
no factor explains.
"""
from __future__ import annotations

import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FACTORS_ZIP = FF_BASE + "F-F_Research_Data_Factors_daily_CSV.zip"
FF_MOMENTUM_ZIP = FF_BASE + "F-F_Momentum_Factor_daily_CSV.zip"

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "ff_factors.parquet"

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "Mom"]


def _fetch_zip_csv(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")


def _parse_ff_csv(text: str) -> pd.DataFrame:
    """The French CSVs have preamble text, a header row, daily rows (YYYYMMDD),
    then sometimes an annual section — keep only the daily rows."""
    rows: list[list[str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if re.fullmatch(r"\d{8}", parts[0] or ""):
            rows.append(parts)
        elif header is None and len(parts) > 1 and not parts[0] and all(parts[1:]):
            header = parts[1:]
    if not rows:
        raise ValueError("No daily rows found in factor file")
    ncols = len(rows[0]) - 1
    if header is None or len(header) != ncols:
        header = [f"F{i}" for i in range(ncols)]
    df = pd.DataFrame(rows, columns=["Date"] + header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float) / 100.0
    return df.where(df > -0.9)  # -99.99 / -999 are missing-value sentinels


def load_ff_factors(max_age_hours: float = 7 * 24) -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, Mom and RF (decimals), with a weekly parquet
    cache and stale-cache fallback if the download fails."""
    if CACHE.exists():
        age = (time.time() - CACHE.stat().st_mtime) / 3600.0
        if age < max_age_hours:
            return pd.read_parquet(CACHE)
    try:
        factors = _parse_ff_csv(_fetch_zip_csv(FF_FACTORS_ZIP))
        mom = _parse_ff_csv(_fetch_zip_csv(FF_MOMENTUM_ZIP))
        mom.columns = ["Mom"]
        df = factors.join(mom, how="inner")
        CACHE.parent.mkdir(exist_ok=True)
        df.to_parquet(CACHE)
        return df
    except Exception:
        if CACHE.exists():
            return pd.read_parquet(CACHE)
        raise


def factor_regression(returns: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of daily excess strategy returns on the four factors.

    Returns annualized alpha with its t-stat, per-factor loadings and t-stats,
    R² and the number of observations.
    """
    df = factors.join(returns.rename("strat"), how="inner").dropna()
    if len(df) < 60:
        raise ValueError("Not enough overlapping observations for a regression")
    y = (df["strat"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[f].to_numpy() for f in FACTOR_NAMES])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    tstats = beta / np.sqrt(np.diag(cov))
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())

    return {
        "alpha_ann": beta[0] * 252,
        "alpha_t": tstats[0],
        "loadings": dict(zip(FACTOR_NAMES, beta[1:])),
        "tstats": dict(zip(FACTOR_NAMES, tstats[1:])),
        "r2": r2,
        "nobs": len(y),
    }
