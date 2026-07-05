"""Single-stock vs benchmark tab: the same signal engine applied to one equity,
judged with benchmark-relative statistics (alpha, beta, information ratio)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import ACCENT, GREY, PRIMARY, RED, style_fig
from data.loader import load_universe
from engine import backtest, metrics, signals

STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "JPM": "JPMorgan",
    "XOM": "ExxonMobil", "JNJ": "Johnson & Johnson",
}
BENCHMARKS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)",
              "IWM": "Russell 2000 (IWM)", "DIA": "Dow Jones (DIA)"}

FORMATS = {
    "CAGR": "{:.2%}", "Ann. Vol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
    "Max Drawdown": "{:.2%}", "Calmar": "{:.2f}", "Ann. Turnover": "{:.1f}x",
    "Hit Rate": "{:.1%}", "Beta": "{:.2f}", "Alpha (ann.)": "{:+.2%}",
    "Tracking Error": "{:.2%}", "Information Ratio": "{:.2f}", "Correlation": "{:.2f}",
}


@st.cache_data(ttl=3600, show_spinner="Downloading stock data…")
def get_pair(stock: str, bench: str, start: str) -> pd.DataFrame:
    return load_universe([stock, bench], start)


def render() -> None:
    st.markdown(
        "The same signal engine, pointed at a **single stock** instead of an asset-class "
        "portfolio — and judged the way professional equity investors judge everything: "
        "**relative to a benchmark**. Beta measures how much of the performance is just "
        "market exposure; **alpha** is what remains after stripping that out — the part a "
        "manager can actually take credit for. Single names are far noisier than indices "
        "(earnings surprises, company-specific news), so this is a much harsher test of "
        "a signal than the diversified portfolio in the first tab."
    )

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            choice = st.selectbox("Stock", list(STOCKS), format_func=lambda t: f"{STOCKS[t]} ({t})")
            custom = st.text_input("…or type any Yahoo Finance ticker", "",
                                   help="E.g. NFLX, ASML, MC.PA (LVMH), 7203.T (Toyota). "
                                        "Overrides the dropdown if filled.")
            ticker = custom.strip().upper() or choice
        with c2:
            bench = st.selectbox("Benchmark", list(BENCHMARKS), format_func=BENCHMARKS.get,
                                 help="The index the stock strategy has to beat — and the "
                                      "yardstick for alpha and beta.")
            signal_type = st.selectbox("Signal", signals.SIGNAL_TYPES, key="stock_signal")
        with c3:
            start_year = st.slider("Start year", 2000, 2022, 2010, key="stock_start",
                                   help="From 2000 the sample includes the dot-com crash, "
                                        "the GFC, COVID and the 2022 bear market.")
            direction = st.radio("Direction", ["Long / Short", "Long / Flat"],
                                 horizontal=True, key="stock_dir")
            cost_bps = st.slider("Transaction cost (bps)", 0.0, 25.0, 5.0, 0.5, key="stock_cost",
                                 help="Single stocks trade with wider spreads than index ETFs — "
                                      "5–10 bps is realistic for large caps.")
            regime_on = st.toggle("Benchmark regime filter (200-day MA)", key="stock_regime",
                                  help="Only allow long positions while the benchmark index is "
                                       "above its own 200-day moving average (and shorts only "
                                       "below it). Single-stock signals are noisy; the index "
                                       "trend is far more reliable — this gates the former "
                                       "with the latter.")

        p1, p2, p3 = st.columns(3)
        params: dict = {}
        if signal_type in ("MA Crossover", "Combined"):
            with p1:
                params["short"] = st.slider("Fast MA window (days)", 5, 100, 50, 5, key="stock_fast")
            with p2:
                params["long"] = st.slider("Slow MA window (days)", 100, 300, 200, 10, key="stock_slow")
        if signal_type in ("Time-Series Momentum", "Combined"):
            with p3:
                params["lookback"] = st.slider("Momentum lookback (days)", 21, 252, 126, 21,
                                               key="stock_mom")
        if signal_type == "Mean Reversion (Z-Score)":
            with p1:
                params["lookback"] = st.slider("Z-score lookback (days)", 5, 100, 20, 5, key="stock_mrlb")
            with p2:
                params["z_entry"] = st.slider("Entry threshold (std devs)", 0.5, 2.5, 1.0, 0.25,
                                              key="stock_mrz")
        if signal_type == "Ichimoku Cloud":
            with p1:
                params["conversion"] = st.slider("Conversion line window (days)", 5, 20, 9, 1,
                                                 key="stock_ichc")
            with p2:
                params["base"] = st.slider("Base line window (days)", 20, 60, 26, 2,
                                           key="stock_ichb")

    if signal_type in ("MA Crossover", "Combined") and params["short"] >= params["long"]:
        st.error("The fast MA window must be shorter than the slow MA window.")
        return
    if ticker == bench:
        st.error("Pick a stock different from the benchmark.")
        return

    try:
        panel = get_pair(ticker, bench, f"{start_year}-01-01")
    except Exception:
        st.error(f"Could not download data for '{ticker}'. Check the ticker spelling "
                 "(Yahoo Finance format) and try again.")
        return
    if ticker not in panel.columns or panel[ticker].dropna().empty:
        st.error(f"No usable price history for '{ticker}'.")
        return

    # ------------------------------------------------------------- backtest
    long_only = direction == "Long / Flat"
    close = panel[[ticker]]
    sig = signals.build_signal(close, signal_type, long_only=long_only, **params)
    if regime_on:
        sig = signals.regime_filter(sig, panel[bench])
    bt = backtest.run_backtest(close, sig, cost_bps)

    strat = bt.strategy_returns[ticker]
    stock_bh = panel[ticker].pct_change(fill_method=None).fillna(0.0)
    bench_bh = panel[bench].pct_change(fill_method=None).fillna(0.0)

    stats_strat = metrics.summary(strat, positions=bt.positions[ticker], turnover=bt.turnover[ticker])
    rel_strat = metrics.benchmark_relative(strat, bench_bh)

    # ------------------------------------------------------------- headline
    st.markdown(f"#### Strategy on {ticker} vs {bench} — headline metrics")
    cols = st.columns(6)
    headline = [
        ("CAGR", f"{stats_strat['CAGR']:.2%}", "Compound annual growth of the strategy on this stock."),
        ("Sharpe", f"{stats_strat['Sharpe']:.2f}", "Return per unit of total risk."),
        ("Alpha (ann.)", f"{rel_strat['Alpha (ann.)']:+.2%}",
         "Annualized return left after removing the benchmark-driven part. "
         "Positive alpha is the whole point of active management."),
        ("Beta", f"{rel_strat['Beta']:.2f}",
         "Sensitivity to the benchmark: 1 = moves one-for-one with the index, "
         "0 = independent of it. Trend strategies typically show low beta."),
        ("Info Ratio", f"{rel_strat['Information Ratio']:.2f}",
         "Active return per unit of tracking error — the benchmark-relative Sharpe. "
         "Above ~0.5 is considered good among professional managers."),
        ("Max Drawdown", f"{stats_strat['Max Drawdown']:.1%}", "Worst peak-to-trough loss."),
    ]
    for col, (label, value, help_text) in zip(cols, headline):
        col.metric(label, value, help=help_text)

    # ------------------------------------------------------------- charts
    log_scale = st.toggle("Log scale", value=True, key="stock_log")
    eq_strat = metrics.equity_curve(strat)
    eq_stock = metrics.equity_curve(stock_bh)
    eq_bench = metrics.equity_curve(bench_bh)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_bench.index, y=eq_bench, name=f"{bench} Buy & Hold",
                             line=dict(width=1.6, color=GREY, dash="dash")))
    fig.add_trace(go.Scatter(x=eq_stock.index, y=eq_stock, name=f"{ticker} Buy & Hold",
                             line=dict(width=1.6, color=ACCENT)))
    fig.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat, name=f"Strategy on {ticker}",
                             line=dict(width=2.8, color=PRIMARY)))
    style_fig(fig, "Growth of $1 (net of costs)", height=460, y_title="Growth of $1")
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**How to read this:** three ways of deploying the same dollar — passively in the "
        "index (grey), passively in the stock (blue), or timed by the signal (teal). The "
        "interesting comparisons are teal vs blue (does timing beat holding the stock?) and "
        "teal vs grey (does the whole exercise beat just buying the index?)."
    )

    ch_left, ch_right = st.columns([1, 1])
    with ch_left:
        rel_curve = eq_strat / eq_bench
        rf = go.Figure()
        rf.add_trace(go.Scatter(x=rel_curve.index, y=rel_curve, name="Strategy / Benchmark",
                                line=dict(width=2.0, color=PRIMARY)))
        rf.add_hline(y=1.0, line_dash="dash", line_color="#2C3644")
        style_fig(rf, f"Relative strength: strategy vs {bench}", height=360,
                  y_title="Ratio of equity curves")
        st.plotly_chart(rf, width="stretch")
        st.caption(
            "**How to read this:** the strategy's equity divided by the benchmark's. "
            "Rising = outperforming the index, falling = lagging it, flat = just matching "
            "it (in which case an investor would rather own the cheap index fund)."
        )
    with ch_right:
        dd = go.Figure()
        dd.add_trace(go.Scatter(x=strat.index, y=metrics.drawdown_series(strat),
                                name="Strategy", fill="tozeroy",
                                line=dict(color=PRIMARY, width=1.5)))
        dd.add_trace(go.Scatter(x=stock_bh.index, y=metrics.drawdown_series(stock_bh),
                                name=f"{ticker} Buy & Hold",
                                line=dict(color=RED, width=1.2, dash="dot")))
        style_fig(dd, "Drawdown — % below previous peak", height=360, y_title="Drawdown")
        dd.update_yaxes(tickformat=".0%")
        st.plotly_chart(dd, width="stretch")
        st.caption(
            "**How to read this:** single stocks routinely draw down 50%+ (a diversified "
            "index rarely does). If the signal's main achievement is cutting that tail "
            "risk, it can be valuable even with a similar CAGR."
        )

    # ------------------------------------------------------------- table
    st.markdown("#### Full comparison (net of costs)")
    rows = {
        f"Strategy on {ticker}": {**stats_strat, **rel_strat},
        f"{ticker} Buy & Hold": {**metrics.summary(stock_bh),
                                 **metrics.benchmark_relative(stock_bh, bench_bh)},
        f"{bench} Buy & Hold": {**metrics.summary(bench_bh),
                                **metrics.benchmark_relative(bench_bh, bench_bh)},
    }
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format(FORMATS), width="stretch")
    st.caption(
        "**How to read this:** the buy-and-hold stock row usually shows beta near 1 and "
        "some alpha (the stock's own out/under-performance). A good timed strategy shows "
        "**lower beta** (less market dependence), a **shallower max drawdown**, and — if "
        "the signal genuinely works on this name — **positive alpha with an information "
        "ratio the stock alone can't match**."
    )
