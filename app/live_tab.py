"""Live forward-track tab: the frozen strategy's out-of-sample record."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import GREY, PRIMARY, style_fig
from data.loader import UNIVERSE, load_universe
from engine import live, metrics

REPO = "https://github.com/octaviodefreitas/multi-asset-research-lab"


@st.cache_data(ttl=3600, show_spinner="Reconstructing the live track from fresh data…")
def get_track():
    config = live.load_config()
    panel = load_universe(config["universe"], config["history_start"])
    return live.live_track(panel, config), config


def render() -> None:
    track, config = get_track()
    freeze, live_r = track["freeze"], track["live"]

    st.markdown(
        f"Backtests can always be tuned after the fact — a **forward track record** cannot. "
        f"On **{freeze:%d %B %Y}** this strategy's parameters were frozen and "
        f"[committed to the public git history]({REPO}/commits/main/live_config.json), where "
        f"they cannot be quietly changed. Everything on this page after that date is the "
        f"strategy trading (on paper) through data that **did not exist when the parameters "
        f"were chosen** — the strongest possible answer to “isn't this curve-fit?”"
    )
    with st.expander("How the track record is verifiable"):
        st.markdown(
            f"""
1. The strategy definition lives in
   [`live_config.json`]({REPO}/blob/main/live_config.json): signal type, parameters,
   universe and cost assumption. The strategy is **fully deterministic** given
   this file.
2. Git records *when* that file was committed. Since the strategy is deterministic,
   recomputing it from fresh market data reproduces exactly what a live run would
   have done since that date — no database needed, and nothing to take on trust.
3. Anyone can clone the repository, check the commit date, run the code and get
   the same numbers. Changing the parameters would show up in the file's
   [commit history]({REPO}/commits/main/live_config.json).

**Frozen settings:** {config["signal_type"]} signal
({", ".join(f"{k}={v}" for k, v in config["params"].items())}),
long/short, equal-weight across {len(config["universe"])} assets,
{config["cost_bps"]:.0f} bps transaction costs.
            """
        )

    # ------------------------------------------------------------- headline
    days_live = len(live_r)
    ret_live = (1.0 + live_r).prod() - 1.0 if days_live else 0.0
    sharpe_live = metrics.sharpe_ratio(live_r) if days_live >= 21 else None
    dd_live = metrics.max_drawdown(live_r) if days_live else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading days live", f"{days_live}",
              help="Days elapsed since the parameter freeze. The record grows automatically.")
    c2.metric("Return since freeze", f"{ret_live:+.2%}",
              help="Cumulative net portfolio return over the live period.")
    c3.metric("Live Sharpe (ann.)", f"{sharpe_live:.2f}" if sharpe_live is not None else "—",
              help="Shown once at least one month of live data exists — annualized "
                   "ratios on a handful of days are statistically meaningless.")
    c4.metric("Live max drawdown", f"{dd_live:.2%}",
              help="Worst peak-to-trough loss within the live period.")

    if days_live < 5:
        st.info(
            f"The track record began accruing on {freeze:%d %B %Y}. Early days — every "
            "market day from here on adds to it automatically. Check back over the "
            "coming weeks and months."
        )

    # ------------------------------------------------------------- chart
    port = track["portfolio"]
    eq = metrics.equity_curve(port)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Backtest (pre-freeze)",
                             line=dict(width=1.4, color=GREY)))
    live_eq = eq.loc[eq.index > freeze]
    if len(live_eq):
        fig.add_trace(go.Scatter(x=live_eq.index, y=live_eq, name="Live (post-freeze)",
                                 line=dict(width=2.8, color=PRIMARY)))
        fig.add_vrect(x0=freeze, x1=eq.index[-1], fillcolor=PRIMARY,
                      opacity=0.06, line_width=0)
    fig.add_vline(x=freeze, line_dash="dash", line_color=PRIMARY,
                  annotation_text=" parameters frozen", annotation_font_color=PRIMARY)
    style_fig(fig, "Growth of $1 — grey is backtest, teal is the live out-of-sample record",
              height=460, y_title="Growth of $1")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**How to read this:** everything left of the dashed line is a backtest and should "
        "be discounted accordingly. Everything right of it was produced on data that did not "
        "exist when the strategy was fixed — the only part that deserves full trust. If the "
        "live segment behaves roughly like the backtest (similar volatility and drawdowns, "
        "not necessarily identical returns), the research process is doing its job."
    )

    if len(live_r) >= 2:
        live_only = go.Figure()
        live_only.add_trace(go.Scatter(x=live_eq.index, y=live_eq / live_eq.iloc[0],
                                       name="Live equity", fill="tozeroy",
                                       line=dict(width=2.4, color=PRIMARY)))
        live_only.add_hline(y=1.0, line_dash="dash", line_color="#2C3644")
        style_fig(live_only, "Live period only — growth of $1 since the freeze",
                  height=320, y_title="Growth of $1")
        st.plotly_chart(live_only, width="stretch")

    # ------------------------------------------------------------- positions
    st.markdown("#### What the strategy holds right now")
    pos = track["current_positions"]
    rows = []
    for ticker, value in pos.items():
        stance = "🟢 Long" if value > 0 else ("🔴 Short" if value < 0 else "⚪ Flat")
        rows.append({"Asset": UNIVERSE.get(ticker, ticker), "Stance": stance,
                     "Position (of sleeve)": f"{value:+.0%}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "**How to read this:** the signal computed at the latest close — the position the "
        "strategy carries into the next trading session. Each asset is an equal sleeve of "
        "the portfolio; ±50% means the two component signals (trend and momentum) currently "
        "disagree on that asset, so conviction is halved."
    )
