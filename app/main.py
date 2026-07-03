"""Entry point: streamlit run app/main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import execution_tab, research_tab

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

tab_research, tab_execution = st.tabs(
    ["📊  Signal Research & Backtest", "⚡  Execution Simulation"]
)

with tab_research:
    research_tab.render()

with tab_execution:
    execution_tab.render()

st.markdown(
    "<hr style='border-color:#1F2733'><p style='color:#8B949E; font-size:0.85rem;'>"
    "Data: Yahoo Finance (daily, adjusted). Simulated research only — not investment advice.</p>",
    unsafe_allow_html=True,
)
