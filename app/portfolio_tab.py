"""Portfolio construction & risk tab: weighting schemes, Monte Carlo
projection, and factor-exposure regression — all run on the frozen live
strategy so the analyses describe one coherent object."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import ACCENT, GOLD, GREY, PRIMARY, RED, style_fig
from data.loader import UNIVERSE, load_universe
from engine import backtest, factors, live, metrics, portfolio, risk

SCHEME_COLORS = {"Equal Weight": PRIMARY, "Inverse Volatility": ACCENT,
                 "Mean-Variance (tangency)": GOLD}

FMT = {"CAGR": "{:.2%}", "Ann. Vol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
       "Max Drawdown": "{:.2%}", "Calmar": "{:.2f}", "Ann. Turnover": "{:.1f}x",
       "Hit Rate": "{:.1%}"}


@st.cache_data(ttl=3600, show_spinner="Running the frozen strategy…")
def get_strategy_returns():
    config = live.load_config()
    panel = load_universe(config["universe"], config["history_start"])
    track = live.live_track(panel, config)
    strat = track["backtest"].strategy_returns.where(panel.notna())
    return strat, config


def render() -> None:
    strat, config = get_strategy_returns()
    st.markdown(
        "Signal research answers *what to trade*; this tab answers **how much of each** — "
        "and what the resulting portfolio's risk actually looks like. All analyses below "
        "run on the frozen strategy from the *Live Forward Track* tab, so every number "
        "describes the same object."
    )

    # ============================================================= weighting
    st.markdown("#### 1 · How should the six sleeves be weighted?")
    st.markdown(
        "Three classic answers, from naive to 'optimal': **equal weight** (1/N, no "
        "estimation at all), **inverse volatility** (risk parity's core idea — every asset "
        "contributes similar risk), and **mean-variance** (Markowitz's tangency portfolio, "
        "re-estimated monthly from trailing data). The punchline of decades of research: "
        "the 'optimal' one often loses to 1/N out of sample, because it trusts noisy "
        "return forecasts."
    )
    schemes = {
        "Equal Weight": portfolio.equal_weights(strat),
        "Inverse Volatility": portfolio.inverse_vol_weights(strat),
        "Mean-Variance (tangency)": portfolio.tangency_weights(strat),
    }
    rets = {name: portfolio.apply_weights(strat, w) for name, w in schemes.items()}

    fig = go.Figure()
    for name, r in rets.items():
        eq = metrics.equity_curve(r)
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                 line=dict(width=2.0, color=SCHEME_COLORS[name])))
    style_fig(fig, "Growth of $1 by weighting scheme (same underlying signals)",
              height=420, y_title="Growth of $1")
    st.plotly_chart(fig, width="stretch")

    tbl = pd.DataFrame({name: metrics.summary(r) for name, r in rets.items()}).T
    st.dataframe(tbl.drop(columns=["Ann. Turnover", "Hit Rate"]).style.format(FMT),
                 width="stretch")

    latest = pd.DataFrame({name: w.iloc[-1] for name, w in schemes.items()})
    latest.index = [UNIVERSE.get(t, t) for t in latest.index]
    wfig = go.Figure()
    for name in schemes:
        wfig.add_trace(go.Bar(x=latest.index, y=latest[name], name=name,
                              marker_color=SCHEME_COLORS[name]))
    style_fig(wfig, "Current weights under each scheme", height=340, y_title="Weight")
    wfig.update_yaxes(tickformat=".0%")
    wfig.update_layout(hovermode="closest")
    st.plotly_chart(wfig, width="stretch")
    st.caption(
        "**How to read this:** all three portfolios trade the *same* signals — only the "
        "sizing differs. Inverse-vol shrinks the wild assets (Bitcoin, oil) and grows the "
        "calm ones (bonds), usually smoothing the ride. Mean-variance concentrates into "
        "whatever recently looked best — elegant in theory, fragile in practice. If 1/N "
        "keeps up with the clever schemes, that is not a bug; it is one of the most "
        "replicated findings in portfolio research (DeMiguel et al., 2009)."
    )

    # ============================================================= monte carlo
    st.markdown("#### 2 · Monte Carlo — the range of outcomes luck alone allows")
    st.markdown(
        "One equity curve is a single draw from a distribution. Resampling blocks of the "
        "strategy's own history thousands of times shows the **range** of next-12-month "
        "outcomes consistent with its behavior — and how bad the unlucky draws get."
    )
    n_paths = st.select_slider("Simulated paths", [500, 1000, 2500, 5000], 1000)
    ew = rets["Equal Weight"]
    paths = risk.bootstrap_paths(ew, horizon=252, n_paths=int(n_paths), block=21)
    fan = risk.fan_percentiles(paths)

    ffig = go.Figure()
    ffig.add_trace(go.Scatter(x=fan.index, y=fan["p95"], name="95th percentile",
                              line=dict(width=0.5, color=PRIMARY), showlegend=False))
    ffig.add_trace(go.Scatter(x=fan.index, y=fan["p5"], name="5th–95th percentile band",
                              fill="tonexty", fillcolor="rgba(0,212,170,0.12)",
                              line=dict(width=0.5, color=PRIMARY)))
    ffig.add_trace(go.Scatter(x=fan.index, y=fan["p75"], name="25th–75th band",
                              line=dict(width=0), showlegend=False))
    ffig.add_trace(go.Scatter(x=fan.index, y=fan["p25"], name="25th–75th percentile band",
                              fill="tonexty", fillcolor="rgba(0,212,170,0.22)",
                              line=dict(width=0)))
    ffig.add_trace(go.Scatter(x=fan.index, y=fan["p50"], name="Median",
                              line=dict(width=2.2, color=PRIMARY)))
    ffig.add_hline(y=1.0, line_dash="dash", line_color="#2C3644")
    style_fig(ffig, "Simulated growth of $1 over the next 252 trading days",
              height=420, x_title="Trading days ahead", y_title="Growth of $1")
    st.plotly_chart(ffig, width="stretch")

    final = paths[:, -1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median 1-year outcome", f"{np.median(final) - 1:+.1%}",
              help="Half the simulated years end better than this, half worse.")
    m2.metric("5th percentile (bad year)", f"{np.percentile(final, 5) - 1:+.1%}",
              help="1-in-20 downside: only 5% of simulated years end below this.")
    m3.metric("95th percentile (great year)", f"{np.percentile(final, 95) - 1:+.1%}",
              help="1-in-20 upside.")
    m4.metric("P(losing year)", f"{(final < 1.0).mean():.0%}",
              help="Share of simulated 12-month periods that end in a loss.")
    st.caption(
        "**How to read this:** the fan is built by shuffling month-long blocks of the "
        "strategy's real history (block bootstrap, preserving volatility clustering). "
        "The width of the band *is* the honest uncertainty: anyone who shows you a single "
        "projected line is hiding it. Note the band assumes the future resembles the "
        "past — regime changes can and do fall outside it."
    )

    # ============================================================= factors
    st.markdown("#### 3 · Factor exposure — is it alpha, or repackaged beta?")
    st.markdown(
        "Allocators' first question about any strategy: are these returns just paid "
        "compensation for known risk factors (market, size, value, momentum — cheaply "
        "available in index products), or genuine **alpha**? A regression on the daily "
        "[Fama-French factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) "
        "answers it."
    )
    try:
        ff = factors.load_ff_factors()
        reg = factors.factor_regression(ew, ff)
    except Exception:
        st.info("Factor data (Ken French library) is temporarily unavailable — "
                "this section will reappear when the source responds.")
        return

    f1, f2, f3 = st.columns(3)
    f1.metric("Alpha (annualized)", f"{reg['alpha_ann']:+.2%}",
              help="Return left over after removing all four factor exposures.")
    f2.metric("Alpha t-stat", f"{reg['alpha_t']:.2f}",
              help="Statistical confidence in the alpha. |t| > 2 is the conventional "
                   "bar for 'probably not luck'; below that, treat the alpha as noise.")
    f3.metric("R² (variance explained)", f"{reg['r2']:.1%}",
              help="How much of the strategy's day-to-day movement the four factors "
                   "explain. Low R² = the strategy marches to its own drum.")

    names = list(reg["loadings"])
    betas = [reg["loadings"][n] for n in names]
    tstats = [reg["tstats"][n] for n in names]
    bfig = go.Figure(go.Bar(
        x=names, y=betas,
        marker_color=[PRIMARY if abs(t) > 2 else "#2C3644" for t in tstats],
        text=[f"β={b:.2f}<br>t={t:.1f}" for b, t in zip(betas, tstats)],
        textposition="outside",
    ))
    bfig.add_hline(y=0, line_color="#2C3644")
    style_fig(bfig, "Factor loadings (teal = statistically significant, |t| > 2)",
              height=380, y_title="Loading (β)")
    bfig.update_layout(showlegend=False, hovermode="closest")
    st.plotly_chart(bfig, width="stretch")
    st.caption(
        "**How to read this:** each bar is the strategy's sensitivity to one factor — "
        "Mkt-RF is the equity market itself, SMB small-vs-big stocks, HML value-vs-growth, "
        "Mom cross-sectional momentum. Grey bars are statistically indistinguishable from "
        "zero. The flattering result for a trend strategy is small loadings, low R² and "
        "positive alpha; the honest caveat is that a low t-stat means the sample cannot "
        "yet distinguish that alpha from luck — which is exactly why the live forward "
        "track exists."
    )
