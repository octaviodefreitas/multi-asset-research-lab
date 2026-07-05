"""Execution algorithm simulator: TWAP, VWAP and POV over a synthetic
intraday session, with square-root temporary market impact, linear permanent
impact, and half-spread cost per fill.

Cost model per bucket i (all quantities signed by `side`, +1 buy / -1 sell):
    temporary impact (bps) = temp_coef × σ_daily × sqrt(q_i / V_i) × 1e4
    permanent impact       = perm_coef × σ_daily × (cum. executed / ADV),
                             shifting the mid path for the rest of the session
    fill_i = mid_i × (1 + side × (half_spread_bps + temp_bps_i) / 1e4)

Implementation shortfall is measured in bps versus the arrival price. When an
algo does not complete inside the horizon (POV in thin volume), the Perold
opportunity cost of the unfilled shares is reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUCKETS_PER_DAY = 78  # 5-minute buckets in a 6.5h US equity session

ALGOS = ("TWAP", "VWAP", "POV")


def u_shaped_profile(n: int) -> np.ndarray:
    """Classic U-shaped intraday volume profile (heavy open/close), normalized."""
    x = np.linspace(0.0, 1.0, n)
    w = 0.6 + 3.4 * (x - 0.5) ** 2
    return w / w.sum()


def twap_schedule(quantity: float, n_buckets: int) -> np.ndarray:
    return np.full(n_buckets, quantity / n_buckets)


def vwap_schedule(quantity: float, volume_profile: np.ndarray) -> np.ndarray:
    """Slices proportional to the *expected* volume profile, as a real VWAP
    engine schedules against the historical curve, not realized volume."""
    return quantity * volume_profile / volume_profile.sum()


def pov_schedule(quantity: float, market_volumes: np.ndarray, participation: float) -> np.ndarray:
    """Trade a fixed fraction of realized market volume until filled.
    May not complete within the horizon, that is the POV trade-off."""
    schedule = np.zeros(len(market_volumes))
    remaining = quantity
    for i, v in enumerate(market_volumes):
        q = min(participation * v, remaining)
        schedule[i] = q
        remaining -= q
        if remaining <= 0:
            break
    return schedule


@dataclass
class FillResult:
    schedule: np.ndarray
    fills: np.ndarray
    mid_path: np.ndarray
    executed: float
    avg_price: float
    fill_rate: float
    is_bps: float           # shortfall on executed shares vs arrival
    spread_bps: float
    temp_impact_bps: float
    timing_bps: float       # drift + permanent impact residual
    opportunity_bps: float  # Perold cost of unfilled shares (at end-of-horizon mid)
    is_total_bps: float     # fill_rate-weighted IS + opportunity cost


def simulate_fills(schedule: np.ndarray, side: int, arrival_price: float,
                   daily_vol: float, adv: float, market_volumes: np.ndarray,
                   half_spread_bps: float, temp_coef: float, perm_coef: float,
                   noise: np.ndarray) -> FillResult:
    """Simulate fills for one schedule against one market path realization.

    `noise` is the standard-normal driver of the exogenous mid path; passing
    the same array to each algo compares them on an identical market.
    """
    n = len(schedule)
    sigma_bucket = daily_vol / np.sqrt(BUCKETS_PER_DAY)
    log_path = np.cumsum(sigma_bucket * noise[:n])
    perm = perm_coef * daily_vol * np.cumsum(schedule) / adv
    mid = arrival_price * np.exp(log_path) * (1.0 + side * perm)

    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(market_volumes[:n] > 0, schedule / market_volumes[:n], 0.0)
    temp_bps = temp_coef * daily_vol * np.sqrt(participation) * 1e4
    fills = mid * (1.0 + side * (half_spread_bps + temp_bps) / 1e4)

    executed = schedule.sum()
    quantity = executed if executed > 0 else np.nan
    weights = schedule / quantity
    avg_price = float((fills * weights).sum())
    is_bps = side * (avg_price / arrival_price - 1.0) * 1e4

    spread = half_spread_bps
    temp = float((temp_bps * weights).sum())
    timing = is_bps - spread - temp

    fill_rate = 1.0  # caller passes target quantity via schedule sums
    return FillResult(schedule, fills, mid, executed, avg_price, fill_rate,
                      is_bps, spread, temp, timing, 0.0, is_bps)


def run_comparison(order_quantity: float, side: int, arrival_price: float,
                   daily_vol: float, adv: float, horizon_buckets: int,
                   participation: float, half_spread_bps: float,
                   temp_coef: float, perm_coef: float,
                   n_sims: int = 500, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Monte Carlo comparison of TWAP / VWAP / POV on shared market paths.

    Returns a long-form DataFrame (one row per sim × algo) and the first
    simulation's paths/fills for charting.
    """
    rng = np.random.default_rng(seed)
    profile = u_shaped_profile(BUCKETS_PER_DAY)[:horizon_buckets]

    rows = []
    sample: dict = {}
    for sim in range(n_sims):
        market_vols = adv * profile * rng.lognormal(mean=-0.02, sigma=0.25, size=horizon_buckets)
        noise = rng.standard_normal(horizon_buckets)

        schedules = {
            "TWAP": twap_schedule(order_quantity, horizon_buckets),
            "VWAP": vwap_schedule(order_quantity, profile),
            "POV": pov_schedule(order_quantity, market_vols, participation),
        }
        for algo, sched in schedules.items():
            res = simulate_fills(sched, side, arrival_price, daily_vol, adv,
                                 market_vols, half_spread_bps, temp_coef,
                                 perm_coef, noise)
            fill_rate = res.executed / order_quantity
            unfilled = 1.0 - fill_rate
            end_mid = res.mid_path[-1]
            opportunity = unfilled * side * (end_mid / arrival_price - 1.0) * 1e4
            is_total = fill_rate * res.is_bps + opportunity
            rows.append({
                "sim": sim, "algo": algo, "is_bps": res.is_bps,
                "spread_bps": res.spread_bps, "temp_impact_bps": res.temp_impact_bps,
                "timing_bps": res.timing_bps, "fill_rate": fill_rate,
                "opportunity_bps": opportunity, "is_total_bps": is_total,
            })
            if sim == 0:
                sample[algo] = res

    return pd.DataFrame(rows), sample


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-algo mean cost decomposition and IS dispersion, in bps."""
    agg = df.groupby("algo").agg(
        mean_is=("is_bps", "mean"), std_is=("is_bps", "std"),
        spread=("spread_bps", "mean"), temp_impact=("temp_impact_bps", "mean"),
        timing=("timing_bps", "mean"), fill_rate=("fill_rate", "mean"),
        opportunity=("opportunity_bps", "mean"), mean_is_total=("is_total_bps", "mean"),
    )
    return agg.reindex([a for a in ALGOS if a in agg.index])
