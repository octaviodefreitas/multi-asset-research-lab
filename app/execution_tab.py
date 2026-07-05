"""Execution algorithm simulation tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import ALGO_COLORS, GREY, style_fig
from data.loader import UNIVERSE, load_prices
from engine import execution

# FX spot has no consolidated volume tape on Yahoo, so all pairs are excluded here.
EXEC_UNIVERSE = {t: label for t, label in UNIVERSE.items() if not t.endswith("=X")}

ALGO_BLURBS = {
    "TWAP": "**TWAP** (Time-Weighted Average Price) slices the order into equal pieces over the horizon. Simple and predictable, but ignores when the market actually trades.",
    "VWAP": "**VWAP** (Volume-Weighted Average Price) follows the intraday volume curve, trading more at the busy open and close. Lower impact per share, since each slice is a smaller share of what's trading.",
    "POV": "**POV** (Percentage of Volume) trades a fixed fraction of realized market volume. Adapts to liquidity in real time, but may not finish if volume is thin, leaving unfilled risk on the table.",
}


@st.cache_data(ttl=3600, show_spinner="Loading asset statistics…")
def get_asset_stats(ticker: str) -> dict:
    df = load_prices(ticker)
    recent = df.tail(252)
    return {
        "price": float(recent["Close"].iloc[-1]),
        "daily_vol": float(recent["Close"].pct_change().std()),
        "adv": float(df["Volume"].tail(60).mean()),
    }


def render() -> None:
    st.markdown(
        "When the strategy rebalances, *how* the trade is executed matters as much as the "
        "signal itself. This simulator takes one rebalancing order and executes it via three "
        "standard algorithms over a synthetic trading session, using a **square-root market "
        "impact model** plus bid/ask spread cost, then compares **implementation shortfall**, "
        "the all-in cost versus the price at the moment the order arrived."
    )
    with st.expander("What are TWAP, VWAP and POV?"):
        for blurb in ALGO_BLURBS.values():
            st.markdown(blurb)
        st.markdown(
            "Costs come from three places: crossing the **bid/ask spread**, **market impact** "
            "(your own order pushes the price, modeled as impact ∝ σ·√(order/volume), the "
            "canonical square-root law), and **timing risk** (the market drifting while you work "
            "the order)."
        )

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.selectbox("Asset to trade", list(EXEC_UNIVERSE),
                                  format_func=EXEC_UNIVERSE.get,
                                  help="FX spot is excluded: it has no consolidated volume tape.")
            side_label = st.radio("Side", ["Buy", "Sell"], horizontal=True)
        with c2:
            order_pct = st.slider("Order size (% of average daily volume)", 0.5, 25.0, 5.0, 0.5,
                                  help="Institutional rebalances typically run 1-10% of ADV. "
                                       "Bigger orders → more impact, non-linearly.")
            horizon_hours = st.slider("Execution horizon (hours)", 0.5, 6.5, 3.0, 0.5,
                                      help="Longer horizons reduce impact but increase timing risk.")
        with c3:
            participation = st.slider("POV participation rate (%)", 1, 30, 10,
                                      help="Fraction of market volume the POV algo consumes each interval.") / 100
            half_spread_bps = st.slider("Half bid/ask spread (bps)", 0.25, 10.0, 2.0, 0.25,
                                        help="Cost of crossing the spread on every fill. "
                                             "~0.5 bps for SPY, several bps for less liquid names.")

        a1, a2, a3 = st.columns(3)
        with a1:
            temp_coef = st.slider("Temporary impact coefficient", 0.1, 2.0, 0.7, 0.1,
                                  help="Scales the square-root impact term. Empirical estimates cluster around 0.5-1.0.")
        with a2:
            perm_coef = st.slider("Permanent impact coefficient", 0.0, 1.0, 0.3, 0.05,
                                  help="Information leakage: how much of your footprint stays in the price.")
        with a3:
            n_sims = st.select_slider("Monte Carlo simulations", [100, 250, 500, 1000, 2000], 500,
                                      help="Each simulation draws a fresh price path and volume "
                                           "realization; all three algos face the identical market.")

    stats = get_asset_stats(ticker)
    horizon_buckets = max(2, round(horizon_hours / 6.5 * execution.BUCKETS_PER_DAY))
    order_qty = order_pct / 100 * stats["adv"]
    side = 1 if side_label == "Buy" else -1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Arrival price", f"${stats['price']:,.2f}",
              help="Last close, the benchmark price the moment the order arrives.")
    m2.metric("Daily volatility", f"{stats['daily_vol']:.2%}",
              help="Realized daily vol over the past year; drives both impact and timing risk.")
    m3.metric("Avg daily volume", f"{stats['adv']:,.0f}",
              help="60-day average daily volume (shares / units).")
    m4.metric("Order size", f"{order_qty:,.0f}",
              help=f"{order_pct:.1f}% of ADV, {side_label.lower()} side.")

    df, sample = execution.run_comparison(
        order_quantity=order_qty, side=side, arrival_price=stats["price"],
        daily_vol=stats["daily_vol"], adv=stats["adv"],
        horizon_buckets=horizon_buckets, participation=participation,
        half_spread_bps=half_spread_bps, temp_coef=temp_coef,
        perm_coef=perm_coef, n_sims=int(n_sims),
    )
    summary = execution.summarize(df)

    # ------------------------------------------------------------- headline comparison
    st.markdown("#### Implementation shortfall vs arrival price")
    left, right = st.columns([1, 1])

    with left:
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=summary.index, y=summary["mean_is"],
            error_y=dict(type="data", array=summary["std_is"], color=GREY),
            marker_color=[ALGO_COLORS[a] for a in summary.index],
            text=[f"{v:.1f} bps" for v in summary["mean_is"]], textposition="outside",
        ))
        style_fig(bar, "Mean shortfall on executed shares (± 1 std)", height=380, y_title="bps")
        bar.update_layout(showlegend=False)
        st.plotly_chart(bar, width="stretch")

    with right:
        box = go.Figure()
        for algo in execution.ALGOS:
            box.add_trace(go.Box(y=df.loc[df["algo"] == algo, "is_bps"], name=algo,
                                 marker_color=ALGO_COLORS[algo], boxmean=True))
        style_fig(box, "Shortfall distribution across simulations", height=380, y_title="bps")
        box.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(box, width="stretch")

    st.caption(
        "**How to read this:** lower is cheaper. The bars show the average all-in cost of the "
        "trade in basis points versus the arrival price; the whiskers and boxes show how much "
        "that cost varies run-to-run (timing risk). A good execution desk trades off a slightly "
        "higher average against a tighter distribution, or vice versa, depending on urgency."
    )

    # ------------------------------------------------------------- decomposition table
    st.markdown("#### Cost decomposition (mean across simulations, bps)")
    display = pd.DataFrame({
        "Mean IS (executed)": summary["mean_is"],
        "IS Std Dev": summary["std_is"],
        "Spread cost": summary["spread"],
        "Temporary impact": summary["temp_impact"],
        "Timing + permanent": summary["timing"],
        "Fill rate": summary["fill_rate"],
        "Opportunity cost (unfilled)": summary["opportunity"],
        "Total IS incl. unfilled": summary["mean_is_total"],
    })
    st.dataframe(
        display.style.format("{:.2f}", subset=display.columns.drop("Fill rate"))
        .format({"Fill rate": "{:.1%}"})
        .background_gradient(subset=["Total IS incl. unfilled"], cmap="RdYlGn_r"),
        width="stretch",
    )
    st.caption(
        "**How to read this:** the shortfall splits into spread paid on every fill, temporary "
        "impact from demanding liquidity, and timing/permanent drift. POV can show a fill rate "
        "below 100%, the opportunity-cost column prices the risk of the shares it failed to "
        "complete inside the horizon."
    )

    # ------------------------------------------------------------- schedules + sample path
    st.markdown("#### How each algorithm works the order (sample simulation)")
    left2, right2 = st.columns([1, 1])
    minutes = np.arange(horizon_buckets) * 5

    with left2:
        sched_fig = go.Figure()
        for algo in execution.ALGOS:
            cum = np.cumsum(sample[algo].schedule) / order_qty * 100
            sched_fig.add_trace(go.Scatter(x=minutes, y=cum, name=algo, mode="lines",
                                           line=dict(width=2.2, color=ALGO_COLORS[algo],
                                                     shape="hv")))
        style_fig(sched_fig, "Cumulative % of order executed", height=380,
                  x_title="Minutes since arrival", y_title="% filled")
        st.plotly_chart(sched_fig, width="stretch")

    with right2:
        path_fig = go.Figure()
        mid = sample["TWAP"].mid_path
        path_fig.add_trace(go.Scatter(x=minutes, y=mid, name="Mid price",
                                      line=dict(color=GREY, width=1.4)))
        path_fig.add_hline(y=stats["price"], line_dash="dash", line_color="#2C3644",
                           annotation_text="arrival", annotation_font_color=GREY)
        for algo in execution.ALGOS:
            res = sample[algo]
            active = res.schedule > 0
            path_fig.add_trace(go.Scatter(
                x=minutes[active], y=res.fills[active], name=f"{algo} fills",
                mode="markers", marker=dict(size=5, color=ALGO_COLORS[algo], opacity=0.75)))
        style_fig(path_fig, "Price path and fill prices", height=380,
                  x_title="Minutes since arrival", y_title="Price")
        path_fig.update_layout(hovermode="closest")
        st.plotly_chart(path_fig, width="stretch")

    st.caption(
        "**How to read this:** the left chart shows each algorithm's pace, TWAP is a straight "
        "line, VWAP leans into the heavy open/close volume, POV follows realized volume and may "
        "flatten out unfilled. On the right, every dot is a fill; buys sit above the mid by the "
        "spread plus the impact of that slice."
    )
