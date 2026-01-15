import math

import pytest

from dashboard import create_app
from params import StrategyParams, PairCointParams, PyramidParams


class StubProvider:
    """Deterministic snapshot provider for tests."""

    def snapshot(self):
        # Provide a minimal but complete snapshot for the dashboard.
        return {
            "portfolio": {
                "AAA": {
                    "bid": 10.0,
                    "ask": 10.2,
                    "last": 10.1,
                    "position": 100,
                    "unrealized": 5.0,
                    "realized": 2.0,
                },
                "BBB": {
                    "bid": 20.0,
                    "ask": 20.2,
                    "last": 20.1,
                    "position": -50,
                    "unrealized": -3.0,
                    "realized": 1.0,
                },
                # ETF leg to keep gross/net calculations robust
                "ETF": {
                    "bid": 50.0,
                    "ask": 50.4,
                    "last": 50.2,
                    "position": 0,
                    "unrealized": 0.0,
                    "realized": 0.0,
                },
            },
            "case": {"status": "ACTIVE", "period": 1, "tick": 1},
            "trader": {"nlv": 1_000_000.0},
        }


def _params() -> StrategyParams:
    """Construct a tiny param set for the dashboard tests."""
    pyramid = PyramidParams(
        entry_levels=(0.5,),
        entry_sizes=(100,),
        exit_levels=(0.1,),
        stop_loss=2.0,
    )
    pair = PairCointParams(
        a="AAA",
        b="BBB",
        c=0.0,
        beta=1.0,
        std=1.0,
        pyramid=pyramid,
        enabled=True,
    )
    return StrategyParams(pair_coint=[pair], etf_nav=None, width={}, allocator=None)


@pytest.fixture
def client():
    app = create_app(provider=StubProvider(), params=_params())
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload == {"status": "ok"}


def test_positions_endpoint_contains_gross_and_pnl(client):
    resp = client.get("/positions")
    assert resp.status_code == 200
    payload = resp.get_json()

    gross = payload["gross_exposure"]
    net = payload["net_exposure"]
    total_pnl = payload["total_pnl"]

    # Gross should include both legs
    assert gross > 0
    # Net should not explode; within a modest bound given stub data
    assert abs(net) < gross
    # PnL aggregates realized + unrealized
    assert math.isclose(total_pnl, 5.0 + 2.0 - 3.0 + 1.0)


def test_strategies_endpoint_reports_spread(client):
    resp = client.get("/strategies")
    assert resp.status_code == 200
    payload = resp.get_json()

    strategies = payload["strategies"]
    assert strategies, "expected at least one strategy in the response"

    first = strategies[0]
    assert "strategy_id" in first
    assert first["spread"] is not None
    assert "action" in first
    assert "strength" in first
    assert "expected_pnl" in first


def test_root_serves_html_dashboard(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type
    body = resp.data.decode("utf-8")
    assert "Trading Dashboard" in body
    assert "/positions" in body
    assert "/strategies" in body
    assert "Strength" in body
    assert "Sharpe" in body
    assert "computeSharpe" in body

