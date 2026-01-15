from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flask import Flask, jsonify

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi
from market import Market, market
from params import StrategyParams, DEFAULT_PARAMS_PATH
from settings import settings
from strategies import PairCointStrategy, EtfNavStrategy
from strategies.base import Order, get_mid


class SnapshotProvider(Protocol):
    """Lightweight interface to supply market snapshots to the dashboard."""

    def snapshot(self) -> dict[str, Any]:
        """Return a dict with portfolio, case, and trader info."""
        ...


class ApiSnapshotProvider:
    """Pulls data from the live RIT API."""

    def __init__(self, api_key: str, api_host: str) -> None:
        self.client = RotmanInteractiveTraderApi(api_key, api_host)

    def snapshot(self) -> dict[str, Any]:
        portfolio = self.client.get_portfolio()
        case = self.client.get_case()
        trader = self.client.get_trader()
        return {"portfolio": portfolio, "case": case, "trader": trader}


@dataclass
class StrategyView:
    strategy_id: str
    state: str
    action: str
    spread: float | None
    expected_pnl: float | None
    reason: str
    orders: list[dict[str, Any]]


class StrategyInspector:
    """Runs strategy logic without touching execution."""

    def __init__(self, params: StrategyParams, mkt: Market = market) -> None:
        self.market = mkt
        self.strategies = []

        for p in params.pair_coint:
            if p.enabled:
                self.strategies.append(PairCointStrategy(p))

        if params.etf_nav and params.etf_nav.enabled:
            self.strategies.append(EtfNavStrategy(params.etf_nav, self.market))

    def evaluate(self, portfolio: dict, case: dict) -> list[StrategyView]:
        views: list[StrategyView] = []

        for strat in self.strategies:
            signal = strat.compute_signal(portfolio, case)
            spread = getattr(strat, "_spread_adj", None)
            orders = [self._order_dict(o, portfolio) for o in signal.orders]
            expected = self._expected_pnl(signal.orders, portfolio)

            views.append(StrategyView(
                strategy_id=strat.strategy_id,
                state=strat.state.value,
                action=signal.action,
                spread=spread,
                expected_pnl=expected,
                reason=signal.reason,
                orders=orders,
            ))

        return views

    def _order_dict(self, order: Order, portfolio: dict) -> dict[str, Any]:
        sec = portfolio.get(order.ticker, {})
        mid = get_mid(sec)
        return {
            "ticker": order.ticker,
            "quantity": order.quantity,
            "side": order.side,
            "limit": order.price,
            "mid": mid,
        }

    def _expected_pnl(self, orders: list[Order], portfolio: dict) -> float | None:
        """Rough edge estimate to mid for the suggested orders."""
        if not orders:
            return 0.0

        pnl = 0.0
        seen_valid = False
        for order in orders:
            sec = portfolio.get(order.ticker, {})
            mid = get_mid(sec)
            if mid <= 0:
                continue

            seen_valid = True
            if order.side == "BUY":
                pnl += (mid - order.price) * order.quantity
            else:
                pnl += (order.price - mid) * order.quantity

        return pnl if seen_valid else None


def summarize_positions(portfolio: dict, mkt: Market = market) -> dict[str, Any]:
    positions = []
    gross = 0.0
    net = 0.0
    pnl = 0.0

    for ticker in mkt.all_tickers:
        sec = portfolio.get(ticker, {})
        pos = sec.get("position", 0) or 0
        bid = sec.get("bid", 0) or 0
        ask = sec.get("ask", 0) or 0
        mid = get_mid(sec)
        unreal = sec.get("unrealized", 0) or 0
        real = sec.get("realized", 0) or 0

        gross += abs(pos * mid)
        net += pos * mid
        pnl += unreal + real

        positions.append({
            "ticker": ticker,
            "position": pos,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "unrealized": unreal,
            "realized": real,
        })

    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "total_pnl": pnl,
        "positions": positions,
    }


def create_app(
    provider: SnapshotProvider | None = None,
    params: StrategyParams | None = None,
) -> Flask:
    snapshot_provider = provider or ApiSnapshotProvider(settings.api_key, settings.api_host)
    params_obj = params or StrategyParams.load(DEFAULT_PARAMS_PATH)
    inspector = StrategyInspector(params_obj)

    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/positions")
    def positions():
        snap = snapshot_provider.snapshot()
        summary = summarize_positions(snap["portfolio"], market)
        return jsonify(summary)

    @app.route("/strategies")
    def strategies():
        snap = snapshot_provider.snapshot()
        case = snap.get("case", {})
        views = inspector.evaluate(snap["portfolio"], case)

        return jsonify({
            "case": case,
            "strategies": [view.__dict__ for view in views],
        })

    @app.route("/")
    def index():
        return jsonify({
            "status": "ok",
            "endpoints": ["/health", "/positions", "/strategies"],
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)

