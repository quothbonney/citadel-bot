import math

from allocator import Allocator, AllocatorConfig, Signal


def _prices() -> dict[str, float]:
    # Simple deterministic prices for unit_gross computations.
    return {"AAA": 100.0, "BBB": 50.0, "CCC": 10.0, "DDD": 10.0, "ETF": 20.0, "IND": 120.0}


def test_edge_scoring_prefers_higher_net_edge_when_sigma_hat_matches():
    """
    If two signals have identical sigma_hat and legs, the allocator should weight
    the higher net-edge signal more (lambda=0 => all weight to best).
    """
    cfg = AllocatorConfig(
        gross_limit=1_000.0,
        net_limit=1_000.0,
        min_threshold=0.0,  # thresholding is done via entry/edge, not this legacy knob
        top_n=6,
        turnover_pct=1.0,  # disable turnover cap for this test
        horizon_bars=10,
        switch_lambda=0.0,
        regime_cutoff=999.0,
    )
    a = Allocator(cfg)

    prices = _prices()
    legs = {"AAA": -1.0}  # +1 unit SHORT spread => short AAA

    # Tick 0: establish last_S, sigma_hat will be 0 so allocator should flatten.
    sigs0 = [
        Signal(name="A", s_dollars=0.0, entry_dollars=1.0, rt_cost_dollars=0.0, legs=legs),
        Signal(name="B", s_dollars=0.0, entry_dollars=5.0, rt_cost_dollars=0.0, legs=legs),
    ]
    pos0, _ = a.allocate(sigs0, prices, current_pos=None)
    assert all(abs(v) < 1e-9 for v in pos0.values())

    # Tick 1: both have same |ΔS| => same sigma_hat, but A has higher net edge.
    sigs1 = [
        Signal(name="A", s_dollars=10.0, entry_dollars=1.0, rt_cost_dollars=0.0, legs=legs),
        Signal(name="B", s_dollars=10.0, entry_dollars=5.0, rt_cost_dollars=0.0, legs=legs),
    ]
    pos1, active = a.allocate(sigs1, prices, current_pos=None)

    # With lambda=0, allocator should allocate entirely to signal A.
    # budget=1000, unit_gross=|(-1)*100|=100 => units=10 => AAA=-10
    assert math.isclose(pos1["AAA"], -10.0, rel_tol=0, abs_tol=1e-9)
    assert "A" in active


def test_switching_penalty_blocks_rotation_when_edge_gain_is_too_small():
    """
    L1 penalty implies that moving weight delta from A to B costs 2*lambda*delta.
    So if edge_B - edge_A <= 2*lambda, optimizer should not rotate off A.
    """
    cfg = AllocatorConfig(
        horizon_bars=10,
        switch_lambda=0.10,
        regime_cutoff=999.0,
    )
    a = Allocator(cfg)

    edges = {"A": 1.0, "B": 1.1}
    wprev = {"A": 1.0, "B": 0.0}

    w = a._optimize_weights(edges=edges, wprev=wprev)
    assert math.isclose(w["A"], 1.0, abs_tol=1e-12)
    assert math.isclose(w["B"], 0.0, abs_tol=1e-12)


def test_switching_penalty_allows_rotation_when_edge_gain_is_large_enough():
    cfg = AllocatorConfig(
        horizon_bars=10,
        switch_lambda=0.10,
        regime_cutoff=999.0,
    )
    a = Allocator(cfg)

    edges = {"A": 1.0, "B": 1.3}  # gain=0.3 > 2*lambda=0.2
    wprev = {"A": 1.0, "B": 0.0}

    w = a._optimize_weights(edges=edges, wprev=wprev)
    assert math.isclose(w["A"], 0.0, abs_tol=1e-12)
    assert math.isclose(w["B"], 1.0, abs_tol=1e-12)


