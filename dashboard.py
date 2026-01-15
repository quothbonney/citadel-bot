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
        # Minimal inline dashboard: no templates, no bundlers, just fetch+render.
        return (
            """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Trading Dashboard</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #0b0b0b; color: #ddd; }
    h1, h2 { color: #eee; }
    table { border-collapse: collapse; margin-bottom: 24px; width: 100%; }
    th, td { border: 1px solid #444; padding: 6px 8px; text-align: left; }
    th { background: #111; }
    .ok { color: #7df57d; }
    .warn { color: #f5d67d; }
    .err { color: #f57d7d; }
    code { color: #9cf; }
  </style>
</head>
<body>
  <h1>Trading Dashboard</h1>
  <div id="status">Loading...</div>

  <h2>Positions</h2>
  <div id="positions"></div>

  <h2>Strategies</h2>
  <div id="strategies"></div>

  <script>
    async function fetchJson(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + " -> " + r.status);
      return await r.json();
    }

    function renderPositions(data) {
      const gross = data.gross_exposure.toFixed(2);
      const net = data.net_exposure.toFixed(2);
      const pnl = data.total_pnl.toFixed(2);
      const rows = data.positions.map(p => `
        <tr>
          <td>${p.ticker}</td>
          <td>${p.position}</td>
          <td>${p.bid?.toFixed ? p.bid.toFixed(2) : p.bid}</td>
          <td>${p.ask?.toFixed ? p.ask.toFixed(2) : p.ask}</td>
          <td>${p.mid?.toFixed ? p.mid.toFixed(2) : p.mid}</td>
          <td>${p.unrealized?.toFixed ? p.unrealized.toFixed(2) : p.unrealized}</td>
          <td>${p.realized?.toFixed ? p.realized.toFixed(2) : p.realized}</td>
        </tr>
      `).join("");
      return `
        <div>Gross: <span class="${Math.abs(net) <= Math.abs(gross) ? 'ok' : 'warn'}">${gross}</span>
             | Net: <span class="${Math.abs(net) < Math.abs(gross) ? 'ok' : 'warn'}">${net}</span>
             | PnL: <span class="${pnl >= 0 ? 'ok' : 'err'}">${pnl}</span></div>
        <table>
          <thead><tr><th>Ticker</th><th>Pos</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Unreal</th><th>Real</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    function renderStrategies(data) {
      const rows = data.strategies.map(s => `
        <tr>
          <td>${s.strategy_id}</td>
          <td>${s.state}</td>
          <td>${s.action}</td>
          <td>${s.spread === null ? "" : s.spread.toFixed ? s.spread.toFixed(4) : s.spread}</td>
          <td>${s.expected_pnl === null ? "" : s.expected_pnl.toFixed ? s.expected_pnl.toFixed(2) : s.expected_pnl}</td>
          <td>${s.reason}</td>
        </tr>
      `).join("");
      return `
        <div>Case: ${JSON.stringify(data.case)}</div>
        <table>
          <thead><tr><th>ID</th><th>State</th><th>Action</th><th>Spread</th><th>ExpPnL</th><th>Reason</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    async function refresh() {
      try {
        const [pos, strat] = await Promise.all([fetchJson("/positions"), fetchJson("/strategies")]);
        document.getElementById("status").textContent = "OK";
        document.getElementById("positions").innerHTML = renderPositions(pos);
        document.getElementById("strategies").innerHTML = renderStrategies(strat);
      } catch (e) {
        document.getElementById("status").innerHTML = '<span class="err">' + e + '</span>';
      }
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
            """,
            200,
            {"Content-Type": "text/html"},
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)

