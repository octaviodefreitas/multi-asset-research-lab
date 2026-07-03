"""Signal research & backtest tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import ASSET_COLORS, GREY, PRIMARY, RED, style_fig
from data.loader import UNIVERSE, load_universe
from engine import backtest, metrics, signals

# Parameter grids searched in-sample by the walk-forward optimizer.
WF_GRIDS = {
    "MA Crossover": [{"short": s, "long": l}
                     for s, l in [(10, 50), (10, 100), (20, 100), (20, 200), (50, 150), (50, 200)]],
    "Time-Series Momentum": [{"lookback": lb} for lb in (21, 63, 126, 252)],
    "Combined": [{"short": s, "long": l, "lookback": lb}
                 for s, l in [(20, 100), (50, 200)] for lb in (63, 126, 252)],
}

METRIC_FORMATS = {
    "CAGR": "{:.2%}", "Ann. Vol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
    "Max Drawdown": "{:.2%}", "Calmar": "{:.2f}", "Ann. Turnover": "{:.1f}x", "Hit Rate": "{:.1%}",
}


@st.cache_data(ttl=3600, show_spinner="Downloading market data (cached locally after first load)…")
def get_panel(tickers: tuple, start: str) -> pd.DataFrame:
    return load_universe(list(tickers), start)


def render() -> None:
    st.markdown(
        "Systematic trend signals backtested per asset and combined into an "
        "equal-weight portfolio. Adjust any parameter below — every chart and "
        "metric recomputes instantly, with **no lookahead bias**: a signal seen "
        "at today's close only earns tomorrow's return."
    )

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            tickers = st.multiselect(
                "Asset universe", options=list(UNIVERSE),
                default=list(UNIVERSE), format_func=UNIVERSE.get,
                help="Each asset is traded independently by the signal, then combined equal-weight.",
            )
            start_year = st.slider("Backtest start year", 2006, 2022, 2010,
                                   help="Assets that launched later (e.g. Bitcoin, 2014) enter when their data begins.")
        with c2:
            signal_type = st.selectbox("Signal", signals.SIGNAL_TYPES,
                                       help="MA Crossover: fast vs slow moving average. "
                                            "Momentum: sign of the trailing return. "
                                            "Combined: average of the two.")
            direction = st.radio("Direction", ["Long / Short", "Long / Flat"], horizontal=True,
                                 help="Long/Flat never shorts — it goes to cash on a sell signal.")
        with c3:
            cost_bps = st.slider("Transaction cost (bps per unit traded)", 0.0, 25.0, 5.0, 0.5,
                                 help="Charged on every position change: |Δposition| × cost. "
                                      "5 bps ≈ realistic all-in cost for liquid ETFs.")
            vol_target_on = st.toggle("Volatility targeting overlay",
                                      help="Scales positions so each asset targets constant risk — "
                                           "sizes up in calm markets, down in turbulent ones (max 2× leverage).")

        p1, p2, p3, p4 = st.columns(4)
        params: dict = {}
        if signal_type in ("MA Crossover", "Combined"):
            with p1:
                params["short"] = st.slider("Fast MA window (days)", 5, 100, 50, 5)
            with p2:
                params["long"] = st.slider("Slow MA window (days)", 100, 300, 200, 10,
                                           help="50 / 200 is the classic “golden cross”.")
        if signal_type in ("Time-Series Momentum", "Combined"):
            with p3:
                params["lookback"] = st.slider("Momentum lookback (days)", 21, 252, 126, 21)
        target_vol = 0.10
        if vol_target_on:
            with p4:
                target_vol = st.slider("Target volatility (ann.)", 0.05, 0.25, 0.10, 0.01,
                                       format="%.2f")

    if not tickers:
        st.info("Select at least one asset to run the backtest.")
        return
    if signal_type in ("MA Crossover", "Combined") and params["short"] >= params["long"]:
        st.error("The fast MA window must be shorter than the slow MA window.")
        return

    long_only = direction == "Long / Flat"
    panel = get_panel(tuple(tickers), f"{start_year}-01-01")
    valid = panel.notna()

    # ------------------------------------------------------------- backtest
    sig = signals.build_signal(panel, signal_type, long_only=long_only,
                               vol_target_on=vol_target_on, target_vol=target_vol,
                               **params)
    bt = backtest.run_backtest(panel, sig, cost_bps)
    port = backtest.equal_weight_returns(bt.strategy_returns, valid)
    bench = backtest.equal_weight_returns(bt.asset_returns.where(valid), valid)

    port_stats = metrics.summary(port, positions=bt.positions.abs().sum(axis=1),
                                 turnover=bt.turnover.sum(axis=1) / max(len(tickers), 1))

    # ------------------------------------------------------------- headline
    st.markdown("#### Equal-weight portfolio — headline metrics")
    cols = st.columns(6)
    headline = [
        ("CAGR", f"{port_stats['CAGR']:.2%}", "Compound annual growth rate of the strategy."),
        ("Sharpe", f"{port_stats['Sharpe']:.2f}", "Return per unit of risk. Above ~0.8 is respectable for a simple daily strategy."),
        ("Sortino", f"{port_stats['Sortino']:.2f}", "Like Sharpe but only penalizes downside moves."),
        ("Max Drawdown", f"{port_stats['Max Drawdown']:.1%}", "Worst peak-to-trough loss an investor would have endured."),
        ("Calmar", f"{port_stats['Calmar']:.2f}", "CAGR divided by max drawdown — growth per unit of pain."),
        ("Hit Rate", f"{port_stats['Hit Rate']:.1%}", "Share of invested days that made money."),
    ]
    for col, (label, value, help_text) in zip(cols, headline):
        col.metric(label, value, help=help_text)

    # ------------------------------------------------------------- charts
    log_scale = st.toggle("Log scale", value=True,
                          help="Log scale shows compounding fairly — equal vertical distances are equal % moves.")

    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=bt.equity.index, y=bt.equity[t], name=UNIVERSE[t],
                                 line=dict(width=1.1, color=ASSET_COLORS.get(t)), opacity=0.55))
    port_eq = metrics.equity_curve(port)
    bench_eq = metrics.equity_curve(bench)
    fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, name="Buy & Hold (EW)",
                             line=dict(width=1.6, color=GREY, dash="dash")))
    fig.add_trace(go.Scatter(x=port_eq.index, y=port_eq, name="Strategy — EW Portfolio",
                             line=dict(width=2.8, color=PRIMARY)))
    style_fig(fig, "Equity curves — growth of $1 (net of costs)", height=480,
              y_title="Growth of $1")
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**How to read this:** thin lines are the signal applied to each asset on its own; "
        "the bold teal line is the equal-weight portfolio of all of them, and the grey dashed "
        "line is simply holding the same assets with no signal. The gap between teal and grey "
        "is what the signal (and diversification across uncorrelated trends) adds."
    )

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=port.index, y=metrics.drawdown_series(port), name="Strategy",
                                fill="tozeroy", line=dict(color=PRIMARY, width=1.5)))
    dd_fig.add_trace(go.Scatter(x=bench.index, y=metrics.drawdown_series(bench), name="Buy & Hold",
                                line=dict(color=RED, width=1.2, dash="dot")))
    style_fig(dd_fig, "Drawdown — % below the previous peak", height=300, y_title="Drawdown")
    dd_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(dd_fig, width="stretch")
    st.caption(
        "**How to read this:** every dip shows how far the portfolio was underwater versus its "
        "prior high. Shallower and shorter dips than buy & hold is the main practical benefit "
        "of trend-following."
    )

    # ------------------------------------------------------------- per-asset table
    st.markdown("#### Per-asset breakdown (net of costs)")
    rows = {}
    for t in tickers:
        rows[UNIVERSE[t]] = metrics.summary(
            bt.strategy_returns[t], positions=bt.positions[t], turnover=bt.turnover[t])
    rows["EW Portfolio"] = port_stats
    rows["Buy & Hold (EW)"] = metrics.summary(bench)
    table = pd.DataFrame(rows).T
    st.dataframe(
        table.style.format({k: v for k, v in METRIC_FORMATS.items()})
        .background_gradient(subset=["Sharpe"], cmap="Greens", vmin=-0.5, vmax=1.5),
        width="stretch",
    )

    # ------------------------------------------------------------- walk-forward
    st.markdown("#### Walk-forward validation — is this curve-fit?")
    st.markdown(
        "A single backtest with hand-picked parameters can always be tuned until it looks good. "
        "Walk-forward validation is the honest test: the sample is split into consecutive periods, "
        "parameters are chosen **only on past data**, then applied to the next unseen period. "
        "The stitched out-of-sample (OOS) curve below is what a systematic process would actually "
        "have delivered."
    )
    n_splits = st.slider("Number of walk-forward splits", 3, 6, 4,
                         help="More splits = more frequent re-optimization on shorter windows.")

    builder = lambda close, **p: signals.build_signal(
        close, signal_type, long_only=long_only, vol_target_on=vol_target_on,
        target_vol=target_vol, **p)
    wf = backtest.walk_forward(panel, builder, WF_GRIDS[signal_type],
                               n_splits=n_splits, cost_bps=cost_bps)

    oos_eq = metrics.equity_curve(wf.oos_returns)
    wf_fig = go.Figure()
    wf_fig.add_trace(go.Scatter(x=oos_eq.index, y=oos_eq, name="Walk-forward OOS",
                                line=dict(width=2.4, color=PRIMARY)))
    insample_eq = metrics.equity_curve(port.loc[oos_eq.index])
    wf_fig.add_trace(go.Scatter(x=insample_eq.index, y=insample_eq,
                                name="Full-sample (your sliders)",
                                line=dict(width=1.6, color=GREY, dash="dash")))
    for s in wf.splits:
        wf_fig.add_vline(x=s.test_start, line_dash="dot", line_color="#2C3644")
    style_fig(wf_fig, "Out-of-sample equity — parameters re-selected at each dotted line",
              height=380, y_title="Growth of $1")
    st.plotly_chart(wf_fig, width="stretch")

    split_rows = [{
        "Test period": f"{s.test_start:%Y-%m-%d} → {s.test_end:%Y-%m-%d}",
        "Parameters chosen in-sample": ", ".join(f"{k}={v}" for k, v in s.params.items()),
        "In-sample Sharpe": round(s.train_sharpe, 2),
        "Out-of-sample Sharpe": round(s.test_sharpe, 2),
    } for s in wf.splits]
    st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)
    st.caption(
        "**How to read this:** in each row, the parameters were picked using only data *before* "
        "the test period. An out-of-sample Sharpe well below the in-sample one is the signature "
        "of overfitting; roughly comparable values suggest the signal captures something real."
    )
