"""Entry point: streamlit run app/main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import execution_tab, live_tab, portfolio_tab, research_tab, stock_tab

st.set_page_config(
    page_title="Multi-Asset Research & Execution Lab",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <div style="padding: 0 0 0.25rem 0;">
      <h1 style="margin-bottom: 0;">Multi-Asset Research &amp; Execution Lab</h1>
      <p style="color: #8B949E; font-size: 1.05rem; margin-top: 0.25rem;">
        Systematic signal research, walk-forward backtesting and execution-quality
        analysis across equities, bonds, gold, FX, oil and crypto.
      </p>
      <p style="color: #E6EDF3; font-size: 0.95rem; margin-top: 0.35rem;">
        Built by <a href="https://www.linkedin.com/in/octavio-de-freitas"
           style="color: #E6EDF3; text-decoration: none;"><b>Octavio De Freitas</b></a> ·
        <a href="https://github.com/octaviodefreitas/multi-asset-research-lab"
           style="color: #00D4AA; text-decoration: none;">View source on GitHub</a> ·
        Feedback welcome —
        <a href="https://www.linkedin.com/in/octavio-de-freitas"
           style="color: #00D4AA; text-decoration: none;">connect on LinkedIn</a>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "**Disclaimer** — All results shown here are backtested / simulated research "
    "on historical data. They are not live trading performance, involve no real "
    "capital, and do not constitute investment advice. Backtested results are "
    "subject to overfitting and survivorship effects and will differ from live results.",
    icon="⚠️",
)

with st.expander("**About this project — what it is, why it exists, and how to explore it**"):
    st.markdown(
        """
**What this is.** An end-to-end demonstration of how a systematic trading strategy
goes from idea to (simulated) execution — the same workflow a quantitative trading
desk follows: get clean data, build a signal, backtest it honestly, stress-test it
for overfitting, and then measure what it would actually cost to trade.

**Why it exists.** I built this as a portfolio project to demonstrate, in working
code rather than on a CV line, the core skills of quantitative trading research:
data pipelines, vectorized backtesting in pandas, statistical validation
(walk-forward testing, parameter-sensitivity analysis), market microstructure
(implementation shortfall, square-root impact), and communicating results clearly.
The full source code is open on
[GitHub](https://github.com/octaviodefreitas/multi-asset-research-lab), including
unit tests for all of the financial math.

**How to explore it in 60 seconds.**
1. In the **Signal Research & Backtest** tab, pick a signal and drag the sliders —
   every chart and metric recomputes live. The teal line beating (or not beating!)
   the grey dashed buy-and-hold line is the whole story of the strategy.
2. Scroll down to **walk-forward validation** — the honest test, where parameters
   are chosen only on past data and judged on unseen data. Comparing in-sample vs
   out-of-sample Sharpe is how professionals detect overfitting.
3. In the **Single Stock vs Benchmark** tab, point the same engine at any
   individual stock (type any ticker!) and judge it the way equity managers are
   judged — by **alpha, beta and information ratio** against an index.
4. The **Portfolio & Risk** tab covers how much of each asset to hold (equal
   weight vs risk parity vs mean-variance), the honest range of next-year
   outcomes (Monte Carlo), and whether returns are real alpha or repackaged
   factor exposure (Fama-French regression).
5. In the **Execution Simulation** tab, see the part most backtests ignore:
   what the rebalancing trades would actually cost, comparing three standard
   execution algorithms (TWAP, VWAP, POV) across hundreds of simulated market days.
6. The **Live Forward Track** tab is the strongest evidence in the app: the
   strategy's parameters were frozen and committed to public git history, and
   everything after that date is a genuine out-of-sample record that grows
   daily and cannot be retro-fitted.

**No finance background needed** — every chart has a plain-language
*"How to read this"* caption underneath, and every slider has a tooltip (hover
over the small **?** icons) explaining what it controls and what realistic
values look like.
        """
    )

tab_research, tab_stock, tab_portfolio, tab_execution, tab_live = st.tabs([
    "Signal Research & Backtest",
    "Single Stock vs Benchmark",
    "Portfolio & Risk",
    "Execution Simulation",
    "🔴 Live Forward Track",
])

with tab_research:
    research_tab.render()

with tab_stock:
    stock_tab.render()

with tab_portfolio:
    portfolio_tab.render()

with tab_execution:
    execution_tab.render()

with tab_live:
    live_tab.render()

st.markdown(
    "<hr style='border-color:#1F2733'><p style='color:#8B949E; font-size:0.85rem;'>"
    "Built by Octavio De Freitas · "
    "<a href='https://github.com/octaviodefreitas/multi-asset-research-lab' "
    "style='color:#00D4AA; text-decoration:none;'>GitHub</a> · "
    "<a href='https://www.linkedin.com/in/octavio-de-freitas' "
    "style='color:#00D4AA; text-decoration:none;'>LinkedIn</a> — questions and "
    "feedback welcome · "
    "Data: Yahoo Finance (daily, adjusted). Simulated research only — not investment advice.</p>",
    unsafe_allow_html=True,
)
