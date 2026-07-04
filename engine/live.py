"""Live forward track: deterministic reconstruction of the frozen strategy.

The parameters in live_config.json were committed to git on the freeze date.
Because the strategy is fully deterministic given those parameters, recomputing
it daily from fresh market data reproduces exactly what a live run would have
done — and the commit timestamp proves the parameters predate the data. Any
performance after freeze_date is therefore out-of-sample by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine import backtest, signals

CONFIG_PATH = Path(__file__).resolve().parents[1] / "live_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return json.load(fh)


def live_track(panel: pd.DataFrame, config: dict) -> dict:
    """Run the frozen strategy over the full panel and split at the freeze date.

    Returns the full portfolio return series, the live (post-freeze) slice,
    the current positions the strategy holds into the next session, and the
    freeze timestamp.
    """
    sig = signals.build_signal(
        panel, config["signal_type"],
        long_only=(config["direction"] == "long_flat"),
        vol_target_on=config.get("vol_target", False),
        **config["params"],
    )
    bt = backtest.run_backtest(panel, sig, config["cost_bps"])
    port = backtest.equal_weight_returns(bt.strategy_returns, panel.notna())

    freeze = pd.Timestamp(config["freeze_date"])
    live = port.loc[port.index > freeze]
    current_positions = sig.iloc[-1]

    return {"portfolio": port, "live": live, "freeze": freeze,
            "current_positions": current_positions, "backtest": bt}
