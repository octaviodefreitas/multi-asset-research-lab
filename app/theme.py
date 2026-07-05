"""Shared plotly styling so every chart matches the dark Streamlit theme."""
from __future__ import annotations

import plotly.graph_objects as go

PRIMARY = "#00D4AA"
ACCENT = "#5B8DEF"
GOLD = "#F2C94C"
RED = "#EB5757"
GREY = "#8B949E"

# Stable per-asset colors across all charts.
ASSET_COLORS = {
    "SPY": "#5B8DEF",
    "EFA": "#3D6FD4",
    "EEM": "#7FB3F5",
    "AGG": "#9B8AFB",
    "TLT": "#7A6AE0",
    "HYG": "#C29BF2",
    "GLD": "#F2C94C",
    "USO": "#F2994A",
    "DBC": "#D9822B",
    "VNQ": "#6FCF97",
    "EURUSD=X": "#56CCF2",
    "BTC-USD": "#EB5757",
    "EW Portfolio": "#00D4AA",
    "Buy & Hold (EW)": "#8B949E",
}

ALGO_COLORS = {"TWAP": "#5B8DEF", "VWAP": "#00D4AA", "POV": "#F2994A"}


def style_fig(fig: go.Figure, title: str | None = None, height: int = 420,
              y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#11151C",
        font=dict(family="Helvetica, Arial, sans-serif", color="#E6EDF3", size=13),
        title=dict(text=title, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=50, r=25, t=55 if title else 25, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1F2733", zerolinecolor="#2C3644", title=x_title)
    fig.update_yaxes(gridcolor="#1F2733", zerolinecolor="#2C3644", title=y_title)
    return fig
