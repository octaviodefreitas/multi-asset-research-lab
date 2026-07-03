import numpy as np
import pytest

from engine import execution


N = 40
Q = 10_000.0
ADV = 1_000_000.0


def flat_volumes():
    return np.full(N, ADV / N)


def test_twap_and_vwap_schedules_sum_to_order():
    assert execution.twap_schedule(Q, N).sum() == pytest.approx(Q)
    prof = execution.u_shaped_profile(N)
    assert execution.vwap_schedule(Q, prof).sum() == pytest.approx(Q)


def test_vwap_proportional_to_profile():
    prof = execution.u_shaped_profile(N)
    sched = execution.vwap_schedule(Q, prof)
    np.testing.assert_allclose(sched / Q, prof)


def test_pov_respects_participation_cap():
    vols = flat_volumes()
    sched = execution.pov_schedule(Q, vols, participation=0.05)
    assert (sched <= 0.05 * vols + 1e-9).all()
    assert sched.sum() <= Q + 1e-9


def test_pov_incomplete_when_volume_too_thin():
    vols = flat_volumes()
    sched = execution.pov_schedule(Q, vols, participation=0.0001)
    assert sched.sum() < Q  # cannot finish inside the horizon


def _fills(quantity, **overrides):
    kwargs = dict(side=1, arrival_price=100.0, daily_vol=0.01, adv=ADV,
                  market_volumes=flat_volumes(), half_spread_bps=2.0,
                  temp_coef=0.7, perm_coef=0.3, noise=np.zeros(N))
    kwargs.update(overrides)
    return execution.simulate_fills(execution.twap_schedule(quantity, N), **kwargs)


def test_buy_fills_above_mid():
    res = _fills(Q)
    assert (res.fills >= res.mid_path).all()


def test_shortfall_increases_with_order_size():
    """Square-root impact: bigger orders must cost more per share (zero noise)."""
    costs = [_fills(q).is_bps for q in (1_000, 10_000, 100_000)]
    assert costs[0] < costs[1] < costs[2]


def test_cost_decomposition_sums_to_shortfall():
    res = _fills(Q)
    assert res.spread_bps + res.temp_impact_bps + res.timing_bps == pytest.approx(res.is_bps)


def test_comparison_shapes_and_fill_rates():
    df, sample = execution.run_comparison(
        order_quantity=Q, side=1, arrival_price=100.0, daily_vol=0.01, adv=ADV,
        horizon_buckets=N, participation=0.1, half_spread_bps=2.0,
        temp_coef=0.7, perm_coef=0.3, n_sims=20, seed=7)
    assert len(df) == 20 * 3
    assert set(df["algo"]) == set(execution.ALGOS)
    assert ((df["fill_rate"] > 0) & (df["fill_rate"] <= 1.0 + 1e-9)).all()
    assert set(sample) == set(execution.ALGOS)


def test_seed_reproducibility():
    kwargs = dict(order_quantity=Q, side=1, arrival_price=100.0, daily_vol=0.01,
                  adv=ADV, horizon_buckets=N, participation=0.1,
                  half_spread_bps=2.0, temp_coef=0.7, perm_coef=0.3,
                  n_sims=10, seed=123)
    df1, _ = execution.run_comparison(**kwargs)
    df2, _ = execution.run_comparison(**kwargs)
    assert df1.equals(df2)
