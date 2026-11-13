"""Energy check: how do the frozen strategy settings behave on energy vs equities?

Separate research script. It does NOT touch live_config.json or the app universe,
so the frozen live out-of-sample track is unaffected.

Run from the repo root:
    .venv/bin/python energy_check.py

Notes on the data:
- BZ=F (Brent) and NG=F (US Henry Hub gas) are Yahoo "continuous" front-month
  futures: contracts are spliced at each roll without adjustment, so returns
  include roll jumps. Fine for a first look at volatility and trend behaviour,
  not for precise P&L.
- WTI (CL=F) is left out on purpose: its price went negative in April 2020,
  which breaks percentage returns.
- USO is an ETF that holds rolling WTI futures, so it pays the roll cost when the
  curve is in contango. Comparing it with spot-like series is a lesson in itself.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data.loader import load_prices
from engine import metrics
from engine.backtest import run_backtest
from engine.signals import build_signal

PARAMS = {"short": 50, "long": 200, "lookback": 126}  # frozen live settings
COST_BPS = 5.0
START = "2008-01-01"

TICKERS = {
    "SPY": "US Equities (SPY)",
    "GLD": "Gold (GLD)",
    "USO": "Crude Oil ETF (USO)",
    "DBC": "Broad Commodities (DBC)",
    "BZ=F": "Brent Crude futures (BZ=F)",
    "NG=F": "US Natural Gas futures (NG=F)",
}


def load_close() -> pd.DataFrame:
    closes, missing = {}, []
    for t in TICKERS:
        try:
            closes[t] = load_prices(t, START)["Close"]
        except Exception as e:  # noqa: BLE001
            missing.append(f"{t} ({e.__class__.__name__})")
    if missing:
        print("Could not load:", ", ".join(missing), file=sys.stderr)
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[panel.index.dayofweek < 5].ffill(limit=5)
    return panel[panel > 0]


def year_return(r: pd.Series, year: int) -> float:
    r = r[r.index.year == year]
    return float((1 + r).prod() - 1) if len(r) else np.nan


def main() -> None:
    close = load_close()
    rets = close.pct_change(fill_method=None)

    plain = run_backtest(close, build_signal(close, "Combined", long_only=True, **PARAMS), COST_BPS)
    vt = run_backtest(close, build_signal(close, "Combined", long_only=True, vol_target_on=True,
                                          target_vol=0.10, **PARAMS), COST_BPS)

    rows = []
    for t in close.columns:
        live = close[t].notna()
        bh = rets[t][live].dropna()
        st = plain.strategy_returns[t][live]
        pos = plain.positions[t][live]
        vpos = vt.positions[t][live]
        rows.append({
            "Asset": TICKERS[t],
            "From": bh.index[0].date().isoformat(),
            "Vol (ann.)": metrics.annualized_vol(bh),
            "B&H Sharpe": metrics.sharpe_ratio(bh),
            "B&H Max DD": metrics.max_drawdown(bh),
            "Strat Sharpe": metrics.sharpe_ratio(st),
            "Strat Max DD": metrics.max_drawdown(st),
            "Time invested": float((pos > 0).mean()),
            "Avg size w/ 10% vol target": float(vpos[vpos > 0].mean()) if (vpos > 0).any() else np.nan,
            "B&H 2022": year_return(bh, 2022),
            "Strat 2022": year_return(st, 2022),
        })
    table = pd.DataFrame(rows).set_index("Asset")

    pct = ["Vol (ann.)", "B&H Max DD", "Strat Max DD", "Time invested", "B&H 2022", "Strat 2022"]
    fmt = table.copy()
    for c in fmt.columns:
        if c in pct:
            fmt[c] = fmt[c].map(lambda x: f"{x:.0%}" if pd.notna(x) else "n/a")
        elif c != "From":
            fmt[c] = fmt[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")

    lines = [
        "# Energy check: frozen strategy on energy vs equities",
        "",
        f"Signal: Combined (MA {PARAMS['short']}/{PARAMS['long']} + {PARAMS['lookback']}d momentum), "
        f"long/flat, {COST_BPS:.0f} bps costs. Data from {START} (or first available).",
        "",
        fmt.to_markdown(),
        "",
    ]
    if "NG=F" in close.columns:
        ng = rets["NG=F"].dropna()
        winter = ng[ng.index.month.isin([11, 12, 1, 2, 3])]
        summer = ng[ng.index.month.isin([4, 5, 6, 7, 8, 9, 10])]
        lines += [f"Natural gas volatility: winter (Nov-Mar) {metrics.annualized_vol(winter):.0%} "
                  f"vs rest of year {metrics.annualized_vol(summer):.0%}.", ""]
    if {"USO", "BZ=F"} <= set(close.columns):
        both = rets[["USO", "BZ=F"]].dropna()
        lines += [f"Since {both.index[0].year}: Brent front-month series "
                  f"{(1 + both['BZ=F']).prod() - 1:+.0%} vs USO ETF {(1 + both['USO']).prod() - 1:+.0%} "
                  "(gap = mostly roll cost in contango, plus WTI vs Brent).", ""]

    report = "\n".join(lines)
    print(report)
    with open("energy_check_results.md", "w") as f:
        f.write(report)
    print("Saved to energy_check_results.md")


if __name__ == "__main__":
    main()
