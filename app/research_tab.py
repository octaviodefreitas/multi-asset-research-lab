"""Signal research & backtest tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import export
from app.theme import ASSET_COLORS, GREY, PRIMARY, RED, short_label, style_fig
from data.loader import UNIVERSE, load_prices, load_universe
from engine import backtest, metrics, signals

# Parameter grids searched in-sample by the walk-forward optimizer.
WF_GRIDS = {
    "MA Crossover": [{"short": s, "long": l}
                     for s, l in [(10, 50), (10, 100), (20, 100), (20, 200), (50, 150), (50, 200)]],
    "Time-Series Momentum": [{"lookback": lb} for lb in (21, 63, 126, 252)],
    "Mean Reversion (Z-Score)": [{"lookback": lb, "z_entry": z}
                                 for lb in (10, 20, 40, 60) for z in (0.5, 1.0, 1.5)],
    "Ichimoku Cloud": [{"conversion": c, "base": b}
                       for c, b in [(7, 20), (9, 26), (12, 34), (18, 52)]],
    "Combined": [{"short": s, "long": l, "lookback": lb}
                 for s, l in [(20, 100), (50, 200)] for lb in (63, 126, 252)],
}

# Grids scanned by the parameter-sensitivity heatmap.
SENS_SHORTS = [5, 10, 20, 30, 50, 75]
SENS_LONGS = [100, 125, 150, 200, 250, 300]
SENS_MOM_LOOKBACKS = [21, 42, 63, 84, 126, 168, 252]
SENS_MR_LOOKBACKS = [10, 15, 20, 30, 40, 60]
SENS_MR_Z = [0.5, 0.75, 1.0, 1.5, 2.0]
SENS_ICH_CONV = [5, 7, 9, 12, 15, 18]
SENS_ICH_BASE = [20, 26, 34, 44, 52, 60]

# Famous market stress episodes for chart annotation and the crisis table.
EVENTS = [
    ("Dot-com crash", "Dot-com", "2000-03-24", "2002-10-09"),
    ("Global Financial Crisis", "GFC", "2007-10-09", "2009-03-09"),
    ("Eurozone debt crisis", "Euro crisis", "2011-05-02", "2011-10-03"),
    ("China devaluation & oil crash", "2015–16", "2015-08-10", "2016-02-11"),
    ("Q4 2018 selloff", "Q4 '18", "2018-10-01", "2018-12-24"),
    ("COVID crash", "COVID", "2020-02-19", "2020-03-23"),
    ("2022 inflation bear market", "2022 bear", "2022-01-03", "2022-10-12"),
]

METRIC_FORMATS = {
    "CAGR": "{:.2%}", "Ann. Vol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
    "Max Drawdown": "{:.2%}", "Calmar": "{:.2f}", "Ann. Turnover": "{:.1f}x", "Hit Rate": "{:.1%}",
}


@st.cache_data(ttl=3600, show_spinner="Downloading market data (cached locally after first load)…")
def get_panel(tickers: tuple, start: str) -> pd.DataFrame:
    return load_universe(list(tickers), start)


@st.cache_data(show_spinner="Scanning the parameter grid (one backtest per cell)…")
def sensitivity_grid(panel: pd.DataFrame, signal_type: str, long_only: bool,
                     vol_target_on: bool, target_vol: float, cost_bps: float,
                     fixed: dict):
    """Portfolio Sharpe for every parameter combination in a pre-declared grid."""
    valid = panel.notna()

    def sharpe_of(p: dict) -> float:
        sig = signals.build_signal(panel, signal_type, long_only=long_only,
                                   vol_target_on=vol_target_on, target_vol=target_vol, **p)
        bt = backtest.run_backtest(panel, sig, cost_bps)
        return metrics.sharpe_ratio(backtest.equal_weight_returns(bt.strategy_returns, valid))

    if signal_type in ("MA Crossover", "Combined"):
        data = [[sharpe_of({"short": s, "long": l, **fixed}) for l in SENS_LONGS]
                for s in SENS_SHORTS]
        return pd.DataFrame(data, index=SENS_SHORTS, columns=SENS_LONGS)
    if signal_type == "Time-Series Momentum":
        return pd.Series({lb: sharpe_of({"lookback": lb}) for lb in SENS_MOM_LOOKBACKS})
    if signal_type == "Ichimoku Cloud":
        data = [[sharpe_of({"conversion": c, "base": b}) for b in SENS_ICH_BASE]
                for c in SENS_ICH_CONV]
        return pd.DataFrame(data, index=SENS_ICH_CONV, columns=SENS_ICH_BASE)
    data = [[sharpe_of({"lookback": lb, "z_entry": z}) for z in SENS_MR_Z]
            for lb in SENS_MR_LOOKBACKS]
    return pd.DataFrame(data, index=SENS_MR_LOOKBACKS, columns=SENS_MR_Z)


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
                help="12 asset classes: US / intl / EM equities, aggregate / long-duration / "
                     "high-yield bonds, gold / oil / broad commodities, real estate, FX and "
                     "crypto. Each is traded independently by the signal, then combined "
                     "equal-weight.",
            )
            custom_raw = st.text_input(
                "Add any tickers (comma-separated Yahoo symbols)", "",
                help="Extend the universe with anything on Yahoo Finance — e.g. "
                     "NVDA, EWJ, MC.PA, GBPUSD=X.",
            )
            start_year = st.slider("Backtest start year", 2000, 2022, 2010,
                                   help="From 2000 the sample covers the dot-com crash, the GFC, "
                                        "COVID and the 2022 bear. Assets that launched later "
                                        "(e.g. Bitcoin, 2014) enter when their data begins.")
        with c2:
            signal_type = st.selectbox("Signal", signals.SIGNAL_TYPES,
                                       help="MA Crossover: fast vs slow moving average. "
                                            "Momentum: sign of the trailing return. "
                                            "Mean Reversion: fade prices stretched away from "
                                            "their rolling mean (countertrend). "
                                            "Combined: average of the two trend signals.")
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
        if signal_type == "Mean Reversion (Z-Score)":
            with p1:
                params["lookback"] = st.slider("Z-score lookback (days)", 5, 100, 20, 5,
                                               help="Window for the rolling mean and std the price is compared against.")
            with p2:
                params["z_entry"] = st.slider("Entry threshold (std devs)", 0.5, 2.5, 1.0, 0.25,
                                              help="How stretched the price must be before fading it. "
                                                   "Wider bands trade less but with more conviction.")
        if signal_type == "Ichimoku Cloud":
            with p1:
                params["conversion"] = st.slider("Conversion line window (days)", 5, 20, 9, 1,
                                                 help="Tenkan-sen: the fast midpoint line.")
            with p2:
                params["base"] = st.slider("Base line window (days)", 20, 60, 26, 2,
                                           help="Kijun-sen: the slow midpoint line, also the cloud's "
                                                "forward displacement. 9 / 26 / 52 are Hosoda's "
                                                "classic settings; the cloud span is 2× this window.")
        target_vol = 0.10
        if vol_target_on:
            with p4:
                target_vol = st.slider("Target volatility (ann.)", 0.05, 0.25, 0.10, 0.01,
                                       format="%.2f")

    for t in (t.strip().upper() for t in custom_raw.split(",") if t.strip()):
        if t in tickers:
            continue
        try:
            load_prices(t)
            tickers.append(t)
        except Exception:
            st.warning(f"Could not load data for '{t}' — check the Yahoo Finance "
                       "spelling; skipped.")

    if not tickers:
        st.info("Select at least one asset to run the backtest.")
        return
    if signal_type in ("MA Crossover", "Combined") and params["short"] >= params["long"]:
        st.error("The fast MA window must be shorter than the slow MA window.")
        return

    long_only = direction == "Long / Flat"
    panel = get_panel(tuple(tickers), f"{start_year}-01-01")
    valid = panel.notna()

    # Rendered here (top of page), filled at the end once results exist.
    export_slot = st.container(border=True)

    # ------------------------------------------------------------- backtest
    sig = signals.build_signal(panel, signal_type, long_only=long_only,
                               vol_target_on=vol_target_on, target_vol=target_vol,
                               **params)
    bt = backtest.run_backtest(panel, sig, cost_bps)
    port = backtest.equal_weight_returns(bt.strategy_returns, valid)
    bench = backtest.equal_weight_returns(bt.asset_returns.where(valid), valid)

    # Industry-standard reference portfolio: 60/40 equities/bonds, starting
    # once both legs have data.
    bench_6040 = None
    if "SPY" in panel.columns and "AGG" in panel.columns:
        both = valid["SPY"] & valid["AGG"]
        bench_6040 = (0.6 * bt.asset_returns["SPY"]
                      + 0.4 * bt.asset_returns["AGG"]).where(both, 0.0)

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
        fig.add_trace(go.Scatter(x=bt.equity.index, y=bt.equity[t], name=UNIVERSE.get(t, t),
                                 line=dict(width=1.1, color=ASSET_COLORS.get(t)), opacity=0.55))
    port_eq = metrics.equity_curve(port)
    bench_eq = metrics.equity_curve(bench)
    fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, name="Passive EW (rebalanced)",
                             line=dict(width=1.6, color=GREY, dash="dash")))
    if bench_6040 is not None:
        eq_6040 = metrics.equity_curve(bench_6040)
        fig.add_trace(go.Scatter(x=eq_6040.index, y=eq_6040, name="60/40 (SPY/AGG)",
                                 line=dict(width=1.6, color="#C9A227", dash="dot")))
    fig.add_trace(go.Scatter(x=port_eq.index, y=port_eq, name="Strategy — EW Portfolio",
                             line=dict(width=2.8, color=PRIMARY)))
    style_fig(fig, "Equity curves — growth of $1 (net of costs)", height=540,
              y_title="Growth of $1")
    # 12+ legend entries wrap over the title when placed on top; park them below.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(b=120),
    )
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**How to read this:** thin lines are the signal applied to each asset on its own; "
        "the bold teal line is the equal-weight portfolio of all of them. Two benchmarks: "
        "the grey dashed line holds the same assets passively (equal weight, rebalanced — "
        "the 'no skill, same universe' test), and the gold dotted line is the classic 60/40 "
        "equity/bond portfolio — the reference every allocator measures against. Beating "
        "the first shows the signal adds value; beating the second shows the whole approach "
        "earns its place."
    )

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=port.index, y=metrics.drawdown_series(port), name="Strategy",
                                fill="tozeroy", line=dict(color=PRIMARY, width=1.5)))
    dd_fig.add_trace(go.Scatter(x=bench.index, y=metrics.drawdown_series(bench), name="Passive EW",
                                line=dict(color=RED, width=1.2, dash="dot")))
    for _, short_name, ev_start, ev_end in EVENTS:
        s = max(pd.Timestamp(ev_start), port.index[0])
        e = min(pd.Timestamp(ev_end), port.index[-1])
        if s < e:
            dd_fig.add_vrect(x0=s, x1=e, fillcolor=GREY, opacity=0.10, line_width=0,
                             annotation_text=short_name, annotation_position="top left",
                             annotation_font_size=10, annotation_font_color=GREY)
    style_fig(dd_fig, "Drawdown — % below the previous peak (stress episodes shaded)",
              height=320, y_title="Drawdown")
    dd_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(dd_fig, width="stretch")
    st.caption(
        "**How to read this:** every dip shows how far the portfolio was underwater versus its "
        "prior high; the shaded bands mark famous market stress episodes. Shallower and shorter "
        "dips than buy & hold is the main practical benefit of trend-following — but note the "
        "pattern in the bands: slow crises (2008, 2022) suit trend signals, which have time to "
        "flip defensive, while lightning crashes (COVID: −30% in 23 days) hit them before they "
        "can react. The volatility-targeting overlay above is the standard defense — try it."
    )

    crisis_rows = []
    for full_name, _, ev_start, ev_end in EVENTS:
        s, e = pd.Timestamp(ev_start), pd.Timestamp(ev_end)
        if s < port.index[0] or e > port.index[-1]:
            continue
        crisis_rows.append({
            "Episode": full_name,
            "Period": f"{s:%b %Y} – {e:%b %Y}",
            "Strategy": (1 + port.loc[s:e]).prod() - 1,
            "Passive EW": (1 + bench.loc[s:e]).prod() - 1,
        })
    crisis_export = crisis_num = None
    if crisis_rows:
        crisis = pd.DataFrame(crisis_rows)
        crisis["Difference"] = crisis["Strategy"] - crisis["Passive EW"]
        crisis_num = crisis.set_index("Episode")
        crisis_export = crisis.copy()
        for col in ("Strategy", "Passive EW", "Difference"):
            crisis_export[col] = crisis_export[col].map("{:+.1%}".format)
        crisis_export = crisis_export.set_index("Episode")
        st.markdown("##### Crisis playbook — the same episodes, in numbers")
        st.dataframe(
            crisis.style.format({"Strategy": "{:+.1%}", "Passive EW": "{:+.1%}",
                                 "Difference": "{:+.1%}"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "**How to read this:** the strategy's return over each stress episode versus "
            "passively holding the same assets. Positive differences in crises are what "
            "allocators call *crisis alpha* — the main reason institutions pay for "
            "trend-following. Where the difference is negative (fast crashes), that is the "
            "known cost of the approach, not a malfunction; no strategy wins in every regime."
        )

    # ------------------------------------------------------------- tearsheet
    st.markdown("#### Performance through time")
    st.markdown(
        "Headline numbers hide *when* a strategy earns its keep. The heatmap shows every "
        "individual month; the rolling Sharpe shows whether risk-adjusted performance is "
        "stable across market regimes or driven by one lucky stretch."
    )
    t_left, t_right = st.columns([1, 1])

    with t_left:
        monthly = (1.0 + port).resample("ME").prod() - 1.0
        pivot = pd.DataFrame({"r": monthly, "Year": monthly.index.year,
                              "Month": monthly.index.month}).pivot(
            index="Year", columns="Month", values="r")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        mh = go.Figure(go.Heatmap(
            z=pivot.values * 100, x=month_names[:pivot.shape[1]],
            y=pivot.index.astype(str), colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values * 100, 1), texttemplate="%{text}",
            textfont=dict(size=9), colorbar=dict(title="%"),
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
        style_fig(mh, "Monthly returns (%)", height=420)
        mh.update_layout(hovermode="closest")
        mh.update_yaxes(autorange="reversed")
        st.plotly_chart(mh, width="stretch")
        st.caption(
            "**How to read this:** each cell is one calendar month of the portfolio. "
            "You want scattered reds inside mostly green — clusters of deep red rows "
            "reveal regimes where the signal breaks down (e.g. sharp reversals that "
            "whipsaw trend-followers)."
        )

    with t_right:
        window = metrics.TRADING_DAYS
        roll_strat = port.rolling(window).mean() / port.rolling(window).std(ddof=1) * np.sqrt(window)
        roll_bench = bench.rolling(window).mean() / bench.rolling(window).std(ddof=1) * np.sqrt(window)
        rs = go.Figure()
        rs.add_trace(go.Scatter(x=roll_bench.index, y=roll_bench, name="Passive EW",
                                line=dict(color=GREY, width=1.2, dash="dot")))
        rs.add_trace(go.Scatter(x=roll_strat.index, y=roll_strat, name="Strategy",
                                line=dict(color=PRIMARY, width=2.0)))
        rs.add_hline(y=0, line_color="#2C3644")
        style_fig(rs, "Rolling 1-year Sharpe ratio", height=420, y_title="Sharpe")
        st.plotly_chart(rs, width="stretch")
        st.caption(
            "**How to read this:** the Sharpe ratio recomputed over a sliding 1-year "
            "window. A strategy that hugs a stable positive level is far more trustworthy "
            "than one with the same average Sharpe made of wild swings — and stretches "
            "below zero show how long an investor would have had to sit through losses."
        )

    # ------------------------------------------------------------- per-asset table
    st.markdown("#### Per-asset breakdown (net of costs)")
    rows = {}
    for t in tickers:
        rows[UNIVERSE.get(t, t)] = metrics.summary(
            bt.strategy_returns[t], positions=bt.positions[t], turnover=bt.turnover[t])
    rows["EW Portfolio"] = port_stats
    rows["Passive EW (rebalanced)"] = metrics.summary(bench)
    if bench_6040 is not None:
        rows["60/40 (SPY/AGG)"] = metrics.summary(bench_6040)
    table = pd.DataFrame(rows).T
    st.dataframe(
        table.style.format({k: v for k, v in METRIC_FORMATS.items()})
        .background_gradient(subset=["Sharpe"], cmap="Greens", vmin=-0.5, vmax=1.5),
        width="stretch",
    )

    # ------------------------------------------------------------- diversification & robustness
    st.markdown("#### Diversification and parameter robustness")
    st.markdown(
        "Two questions every allocator asks. **Left:** does combining assets actually "
        "diversify — are the per-asset strategies genuinely uncorrelated? **Right:** is the "
        "performance robust to the parameter choice, or did we just find one lucky setting?"
    )
    d_left, d_right = st.columns([1, 1])

    with d_left:
        corr = bt.strategy_returns.where(valid).corr()
        names = [short_label(t) for t in corr.columns]
        ch = go.Figure(go.Heatmap(
            z=corr.values, x=names, y=names,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        style_fig(ch, "Correlation of per-asset strategy returns", height=440)
        ch.update_layout(hovermode="closest")
        ch.update_yaxes(autorange="reversed")
        st.plotly_chart(ch, width="stretch")
        st.caption(
            "**How to read this:** each cell is the correlation between the strategy run on "
            "two different assets (+1 = move together, 0 = independent). The closer the "
            "off-diagonal cells are to zero, the more the equal-weight portfolio smooths "
            "out — six independent return streams cut portfolio volatility far more than "
            "six copies of the same bet. This is the free lunch of diversification."
        )

    with d_right:
        fixed = {"lookback": params["lookback"]} if signal_type == "Combined" else {}
        grid = sensitivity_grid(panel, signal_type, long_only, vol_target_on,
                                target_vol, cost_bps, fixed)
        if isinstance(grid, pd.Series):  # momentum: one parameter -> bar chart
            colors = [PRIMARY if lb == min(grid.index, key=lambda x: abs(x - params["lookback"]))
                      else "#2C3644" for lb in grid.index]
            sh = go.Figure(go.Bar(x=grid.index.astype(str), y=grid.values,
                                  marker_color=colors,
                                  text=np.round(grid.values, 2), textposition="outside"))
            style_fig(sh, "Portfolio Sharpe by momentum lookback (days)", height=440,
                      y_title="Sharpe")
            sh.update_layout(showlegend=False, hovermode="closest")
            st.plotly_chart(sh, width="stretch")
            st.caption(
                "**How to read this:** each bar is a full backtest at a different lookback "
                "(your current setting highlighted). Similar Sharpe across neighbouring "
                "lookbacks means the effect is real; one tall bar surrounded by short ones "
                "means the 'best' parameter is likely noise."
            )
        else:
            if signal_type in ("MA Crossover", "Combined"):
                x_title, y_title_h = "Slow MA window", "Fast MA window"
            elif signal_type == "Ichimoku Cloud":
                x_title, y_title_h = "Base line window", "Conversion line window"
            else:
                x_title, y_title_h = "Entry threshold (std devs)", "Z-score lookback"
            sh = go.Figure(go.Heatmap(
                z=grid.values, x=grid.columns, y=grid.index,
                colorscale="RdYlGn", zmid=0,
                text=np.round(grid.values, 2), texttemplate="%{text}",
                colorbar=dict(title="Sharpe"),
                hovertemplate=f"{y_title_h} %{{y}}, {x_title.lower()} %{{x}}: "
                              "Sharpe %{z:.2f}<extra></extra>"))
            if signal_type in ("MA Crossover", "Combined"):
                cur = (params["long"], params["short"])
            elif signal_type == "Ichimoku Cloud":
                cur = (params["base"], params["conversion"])
            else:
                cur = (params["z_entry"], params["lookback"])
            sh.add_trace(go.Scatter(x=[cur[0]], y=[cur[1]], mode="markers", name="Your setting",
                                    marker=dict(symbol="star", size=15, color="#E6EDF3",
                                                line=dict(color="#0E1117", width=1))))
            style_fig(sh, "Portfolio Sharpe across the parameter grid", height=440,
                      x_title=x_title, y_title=y_title_h)
            sh.update_layout(hovermode="closest", showlegend=False)
            sh.update_xaxes(tickvals=list(grid.columns))
            sh.update_yaxes(tickvals=list(grid.index))
            st.plotly_chart(sh, width="stretch")
            st.caption(
                "**How to read this:** every cell is a complete backtest with those "
                "parameters (★ = your current sliders). A broad plateau of similar green "
                "means the signal works across many settings — evidence it captures "
                "something real. A single bright cell in a sea of red is the classic "
                "fingerprint of an overfit backtest, and walking forward it would likely fail."
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

    # ------------------------------------------------------------- export
    fmt_table = table.copy()
    for col, fmt in METRIC_FORMATS.items():
        fmt_table[col] = fmt_table[col].map(
            lambda v, f=fmt: "—" if pd.isna(v) else f.format(v))

    subtitle = (f"{signal_type} — " + ", ".join(f"{k}={v}" for k, v in params.items())
                + f" | {direction} | {cost_bps:.0f} bps costs | "
                + f"{len(tickers)} assets | {panel.index[0]:%Y}–{panel.index[-1]:%Y}"
                + " | Octavio De Freitas")
    monthly_ds = lambda s: s.resample("ME").last().dropna()
    charts = [
        ("Growth of $1 (net of costs)",
         {"Strategy — EW Portfolio": monthly_ds(port_eq), "Passive EW (rebalanced)": monthly_ds(bench_eq)},
         "0.00"),
        ("Drawdown — % below previous peak",
         {"Strategy": monthly_ds(metrics.drawdown_series(port)),
          "Passive EW": monthly_ds(metrics.drawdown_series(bench))},
         "0%"),
    ]
    export_tables = [("Key metrics (net of costs)", fmt_table)]
    if crisis_export is not None:
        export_tables.append(("Crisis playbook — strategy vs buy & hold", crisis_export))

    workbook_sheets = {
        "Key Metrics": table,
        "Equity Curves": pd.DataFrame({"Strategy (growth of $1)": port_eq,
                                       "Passive EW (growth of $1)": bench_eq}),
        "Daily Returns": pd.DataFrame({"Strategy": port, "Passive EW": bench}),
        "Monthly Returns": pivot,
    }
    if crisis_num is not None:
        workbook_sheets["Crisis Playbook"] = crisis_num

    with export_slot:
        label, b1, b2 = st.columns([2.4, 1.1, 1.0], vertical_alignment="center")
        label.markdown(
            "**Export this configuration** — a presentation-ready tearsheet "
            "(charts are native, editable PowerPoint objects, disclaimer included) "
            "or a full Excel workbook. Both reflect the sliders as currently set."
        )
        b1.download_button(
            "⤓  PowerPoint tearsheet",
            data=export.build_tearsheet(subtitle, charts, export_tables),
            file_name="strategy_tearsheet.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            width="stretch",
            help="Title slide, equity curve, drawdown, key metrics and crisis table — "
                 "drag the slides straight into a pitch book.",
        )
        b2.download_button(
            "⤓  Excel workbook",
            data=export.build_workbook(workbook_sheets),
            file_name="strategy_workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            help="One sheet each: key metrics, equity curves, daily returns, "
                 "monthly returns and the crisis playbook — everything needed to "
                 "rebuild any chart in Excel.",
        )
