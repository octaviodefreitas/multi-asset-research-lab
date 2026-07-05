# Multi-Asset Research & Execution Lab

**Live demo: [octavio-quant-multi-asset-research.streamlit.app](https://octavio-quant-multi-asset-research.streamlit.app/)**
*by Octavio De Freitas*

An interactive research environment for systematic multi-asset trading and
execution-quality analysis, built with Python, pandas and Streamlit.

> ⚠️ **Disclaimer** — Everything in this app is backtested / simulated research on
> historical data. It is **not** live trading performance, involves no real capital,
> and is not investment advice. Backtested results are subject to overfitting and
> will differ from live results.

## What it does

**Tab 1 — Signal Research & Backtest**
- Universe of 14 assets across every major class — US / international /
  emerging-market equities (SPY, EFA, EEM), aggregate / long-duration /
  high-yield bonds (AGG, TLT, HYG), gold / oil / broad commodities (GLD, USO,
  DBC), US real estate (VNQ), three FX pairs (EUR/USD, GBP/USD, USD/JPY) and
  Bitcoin — plus free-text input to add any Yahoo Finance ticker. Daily data
  with a local parquet cache.
- Four signal families — moving-average crossover (incl. the classic 50/200
  golden cross), time-series momentum, z-score mean reversion (countertrend)
  and the Ichimoku cloud — plus an optional volatility-targeting overlay, all
  fully vectorized with pandas.
- **Crisis playbook**: famous stress episodes (GFC, COVID, 2022 bear, ...)
  shaded on the drawdown chart and tabulated strategy-vs-buy-and-hold — the
  "crisis alpha" story allocators actually ask about.
- **No lookahead bias**: a signal observed at the close of day *t* is only
  applied to the return from *t* to *t+1* (and there is a unit test proving it).
- Backtest net of transaction costs, per asset and as an equal-weight portfolio,
  with CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, turnover and hit rate.
- **Walk-forward validation**: parameters are re-selected on expanding in-sample
  windows and evaluated strictly out-of-sample, so the headline curve is not a
  single curve-fit backtest.
- **Robustness analytics**: a parameter-sensitivity Sharpe heatmap (one full
  backtest per grid cell — plateau vs overfit spike), a monthly-returns
  heatmap, a rolling 1-year Sharpe, and a cross-asset strategy correlation
  matrix showing the diversification benefit.

**Tab 2 — Single Stock vs Benchmark**
- The same signal engine applied to any individual equity (dropdown of large
  caps or free-text Yahoo ticker), judged benchmark-relative against a chosen
  index: **alpha, beta, tracking error, information ratio**, relative-strength
  curve and drawdown comparison.

**Tab 3 — Portfolio & Risk**
- Three portfolio-construction schemes on the same underlying signals — equal
  weight, inverse volatility (risk parity) and monthly re-estimated
  mean-variance (tangency) — compared out of sample, all causal.
- **Monte Carlo block bootstrap** of the strategy's history: fan chart of
  next-12-month outcomes, tail percentiles and probability of a losing year.
- **Fama-French factor regression** (Mkt-RF, SMB, HML, Mom, daily data from
  Ken French's library): factor loadings with t-stats, R², and annualized
  alpha — is it real alpha or repackaged factor beta?

**Tab 4 — Execution Simulation**
- Takes a rebalancing order and simulates executing it via **TWAP, VWAP and POV**
  over a configurable intraday horizon.
- Cost model: half bid/ask spread on every fill, **square-root temporary market
  impact** (`cost ∝ σ · √(order / volume)`), linear permanent impact, and a
  U-shaped intraday volume profile.
- Monte Carlo comparison of **implementation shortfall** vs the arrival price —
  mean cost, cost dispersion (timing risk), full decomposition, and fill-rate /
  opportunity-cost accounting when POV fails to complete.

**Tab 5 — Live Forward Track**
- Strategy parameters are frozen in `live_config.json`; the git commit
  timestamp proves they were fixed before the subsequent data existed.
- Because the strategy is deterministic given that file, recomputing it daily
  from fresh data reconstructs exactly what a live run would have done — a
  genuine, verifiable out-of-sample record that grows every trading day,
  with the strategy's current positions shown live.

## Run it locally

```bash
git clone <this-repo>
cd quant-research-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

The first load downloads ~15 years of daily data (a few seconds) and caches it
under `data/cache/`; subsequent runs are instant.

Run the tests:

```bash
pytest
```

## Deploy publicly (free)

The app is stateless and needs no secrets, so it deploys to
[Streamlit Community Cloud](https://share.streamlit.io) unchanged:

1. Push this repo to GitHub (public repo).
2. Go to share.streamlit.io → **New app** → pick the repo, branch `main`,
   main file `app/main.py`.
3. Deploy — you get a public `https://<app>.streamlit.app` URL to share.

## Project structure

```
├── app/                  # Streamlit frontend
│   ├── main.py           #   entry point, tabs, disclaimer
│   ├── research_tab.py   #   signal research & backtest UI
│   ├── stock_tab.py      #   single-stock vs benchmark UI (alpha/beta/IR)
│   ├── portfolio_tab.py  #   weighting schemes, Monte Carlo, factor regression UI
│   ├── execution_tab.py  #   execution simulation UI
│   ├── live_tab.py       #   live forward-track UI
│   └── theme.py          #   shared dark plotly styling
├── data/
│   ├── loader.py         # yfinance download + parquet cache
│   └── cache/            # cached OHLCV (gitignored)
├── engine/
│   ├── signals.py        # MA crossover, momentum, mean reversion, vol targeting
│   ├── backtest.py       # positions, costs, walk-forward validation
│   ├── metrics.py        # CAGR, Sharpe, Sortino, drawdown, alpha/beta/IR, ...
│   ├── portfolio.py      # equal-weight, inverse-vol, tangency weights (causal)
│   ├── risk.py           # block-bootstrap Monte Carlo
│   ├── factors.py        # Fama-French data + OLS factor regression
│   ├── live.py           # frozen-strategy live track reconstruction
│   └── execution.py      # TWAP/VWAP/POV, square-root impact, implementation shortfall
├── live_config.json      # frozen live-strategy definition (git history = proof)
└── tests/                # unit tests for all of the financial math
```

## Methodology notes

- Returns are simple daily returns on adjusted closes; annualization uses 252
  days and a zero risk-free rate.
- Cross-calendar assets (FX, crypto) are aligned to weekdays with a short
  forward-fill; each asset enters the backtest only after its own inception.
- Transaction costs are charged as `|Δposition| × cost` in basis points —
  a reasonable all-in assumption for liquid ETFs at daily frequency.
- The walk-forward optimizer selects parameters by in-sample Sharpe of the
  equal-weight portfolio over a small, pre-declared grid; rolling signals are
  causal, so full-history signal computation introduces no leakage.
- Execution simulation is a stylized model: exogenous lognormal mid-price path,
  U-shaped volume curve with lognormal noise, and the canonical square-root
  impact law. All three algorithms are compared on identical market paths.
