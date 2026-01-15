from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flask import Flask, jsonify

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi
from allocator import Allocator, AllocatorConfig as AllocatorAlgoConfig, Signal as AllocSignal
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
    strength: float | None
    expected_pnl: float | None
    reason: str
    orders: list[dict[str, Any]]


class StrategyInspector:
    """Runs strategy logic without touching execution."""

    def __init__(self, params: StrategyParams, mkt: Market = market) -> None:
        self.market = mkt
        self.params = params
        self.strategies = []
        self.allocator: Allocator | None = None

        for p in params.pair_coint:
            if p.enabled:
                self.strategies.append(PairCointStrategy(p))

        if params.etf_nav and params.etf_nav.enabled:
            self.strategies.append(EtfNavStrategy(params.etf_nav, self.market))

        if params.allocator and params.allocator.enabled:
            cfg = AllocatorAlgoConfig(
                gross_limit=params.allocator.gross_limit,
                net_limit=params.allocator.net_limit,
                max_shares=params.allocator.max_shares or mkt.max_shares,
                top_n=params.allocator.top_n,
                turnover_pct=params.allocator.turnover_pct,
                min_threshold=params.allocator.min_threshold,
                horizon_bars=params.allocator.horizon_bars,
                switch_lambda=params.allocator.switch_lambda,
                regime_cutoff=params.allocator.regime_cutoff,
                w_max=params.allocator.w_max,
                dd_riskoff_enabled=params.allocator.dd_riskoff_enabled,
                dd_riskoff_start=params.allocator.dd_riskoff_start,
                dd_riskoff_full=params.allocator.dd_riskoff_full,
                dd_riskoff_min_scale=params.allocator.dd_riskoff_min_scale,
            )
            self.allocator = Allocator(cfg)

    def evaluate(self, portfolio: dict, case: dict) -> list[StrategyView]:
        views: list[StrategyView] = []

        for strat in self.strategies:
            signal = strat.compute_signal(portfolio, case)
            spread = getattr(strat, "_spread_adj", None)
            orders = [self._order_dict(o, portfolio) for o in signal.orders]
            expected = self._expected_pnl(signal.orders, portfolio)
            strength = self._compute_strength(strat, spread)

            views.append(StrategyView(
                strategy_id=strat.strategy_id,
                state=strat.state.value,
                action=signal.action,
                spread=spread,
                strength=strength,
                expected_pnl=expected,
                reason=signal.reason,
                orders=orders,
            ))

        return views

    def allocator_snapshot(self, portfolio: dict, case: dict) -> dict[str, object] | None:
        """Compute allocator diagnostics for the current snapshot (no execution)."""
        if self.allocator is None:
            return None

        # Build allocator signals (same semantics as runner.py)
        width = self.params.width or {}
        prices = {t: get_mid(portfolio.get(t, {})) for t in self.market.all_tickers}
        current_pos = {t: portfolio.get(t, {}).get("position", 0) for t in self.market.all_tickers}

        sigs: list[AllocSignal] = []
        for strat in self.strategies:
            # compute_signal already called in evaluate(), but call again for safety/standalone usage
            strat.compute_signal(portfolio, case)
            spread = getattr(strat, "_spread_adj", None)
            if spread is None:
                continue

            if isinstance(strat, PairCointStrategy):
                p = strat.params
                pa = prices.get(p.a, 1.0)
                pb = prices.get(p.b, 1.0)
                hb = p.beta * (pa / pb) if pb > 0 else 1.0
                legs = {p.a: -1.0, p.b: hb}
                if p.std <= 0:
                    raise ValueError(f"invalid std for {p.strategy_id}: {p.std}")
                z = float(spread) / float(p.std)
                s_dollars = float(z) * float(pa)
                entry_dollars = float(p.pyramid.first_entry) * float(pa)
                rt_cost_dollars = abs(legs[p.a]) * float(width.get(p.a, 0.0)) + abs(legs[p.b]) * float(width.get(p.b, 0.0))
                sigs.append(AllocSignal(
                    name=strat.strategy_id,
                    s_dollars=s_dollars,
                    entry_dollars=entry_dollars,
                    rt_cost_dollars=rt_cost_dollars,
                    legs=legs,
                ))

            elif isinstance(strat, EtfNavStrategy):
                legs = {"ETF": -1.0, "AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
                rt_cost_dollars = (
                    abs(legs["ETF"]) * float(width.get("ETF", 0.0))
                    + abs(legs["AAA"]) * float(width.get("AAA", 0.0))
                    + abs(legs["BBB"]) * float(width.get("BBB", 0.0))
                    + abs(legs["CCC"]) * float(width.get("CCC", 0.0))
                    + abs(legs["DDD"]) * float(width.get("DDD", 0.0))
                )
                sigs.append(AllocSignal(
                    name=strat.strategy_id,
                    s_dollars=float(spread),
                    entry_dollars=float(strat.params.pyramid.first_entry),
                    rt_cost_dollars=rt_cost_dollars,
                    legs=legs,
                ))

        target_pos, active = self.allocator.allocate(sigs, prices, current_pos)
        orders = self.allocator.to_orders(target_pos, current_pos, prices, min_delta=1)
        diag = self.allocator.diagnostics()

        return {
            "active": active,
            "orders": [o.__dict__ for o in orders],
            "target_pos": target_pos,
            "current_pos": current_pos,
            "diagnostics": diag,
        }

    def _compute_strength(self, strat, spread: float | None) -> float | None:
        """Compute signal strength: |spread| / sigma."""
        if spread is None:
            return None
        
        from strategies.pair_coint import PairCointStrategy
        from strategies.etf_nav import EtfNavStrategy
        
        if isinstance(strat, PairCointStrategy):
            return abs(spread) / strat.params.std if strat.params.std > 0 else None
        elif isinstance(strat, EtfNavStrategy):
            threshold = strat.params.pyramid.first_entry
            return abs(spread) / threshold if threshold > 0 else None
        return None

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
        portfolio = snap["portfolio"]
        views = inspector.evaluate(portfolio, case)
        alloc = inspector.allocator_snapshot(portfolio, case)

        return jsonify({
            "case": case,
            "strategies": [view.__dict__ for view in views],
            "allocator": alloc,
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
    * { box-sizing: border-box; }
    body { font-family: 'Consolas', 'Monaco', monospace; margin: 0; padding: 20px; background: #0a0a0a; color: #ccc; }
    h1 { color: #fff; margin: 0 0 10px 0; font-size: 1.4em; }
    h2 { color: #888; margin: 20px 0 8px 0; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: #111; border: 1px solid #333; padding: 12px; border-radius: 4px; }
    .stats { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 12px; }
    .stat { text-align: center; }
    .stat-value { font-size: 1.4em; font-weight: bold; }
    .stat-label { font-size: 0.75em; color: #666; text-transform: uppercase; }
    table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
    th, td { border: 1px solid #222; padding: 4px 8px; text-align: right; }
    th { background: #1a1a1a; color: #888; font-weight: normal; text-transform: uppercase; font-size: 0.75em; }
    td:first-child, th:first-child { text-align: left; }
    .ok { color: #4f4; }
    .warn { color: #fa0; }
    .err { color: #f44; }
    .active-row { background: #1a2a1a; }
    .flat-row { opacity: 0.6; }
    svg { display: block; width: 100%; height: 120px; }
    .chart-container { margin-top: 8px; }
    #status { font-size: 0.8em; color: #666; }
    .case-info { font-size: 0.85em; color: #888; margin-bottom: 8px; }
    .active-summary { background: #1a2a1a; border: 1px solid #2a4a2a; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <h1>Trading Dashboard <span id="status">...</span></h1>
  <div id="case-info" class="case-info"></div>
  
  <div id="active-summary" class="active-summary" style="display:none;"></div>

  <div class="grid">
    <div class="card">
      <h2>Performance</h2>
      <div id="performance-stats" class="stats"></div>
      <div class="chart-container">
        <svg id="pnl-chart" viewBox="0 0 400 120" preserveAspectRatio="none"></svg>
      </div>
    </div>
    <div class="card">
      <h2>Exposure</h2>
      <div id="exposure-stats" class="stats"></div>
      <div class="chart-container">
        <svg id="exposure-chart" viewBox="0 0 400 120" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </div>

  <h2>Positions</h2>
  <div id="positions"></div>

  <h2>Strategies</h2>
  <h2>Allocator</h2>
  <div id="allocator"></div>
  <div id="strategies"></div>

  <script>
    // Keep ALL returns for cumulative Sharpe, but limit chart points
    const allReturns = [];
    const chartData = { pnl: [], gross: [], net: [] };
    const CHART_POINTS = 60;
    let startPnl = null;

    async function fetchJson(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + " -> " + r.status);
      return await r.json();
    }

    function computeCumulativeSharpe() {
      if (allReturns.length < 5) return null;
      const mean = allReturns.reduce((a, b) => a + b, 0) / allReturns.length;
      const variance = allReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / allReturns.length;
      const std = Math.sqrt(variance);
      if (std < 0.01) return null;
      // Annualize: 2s intervals, ~11700 per trading year
      return (mean / std) * Math.sqrt(11700);
    }

    function fmt(n, decimals = 0) {
      if (n === null || n === undefined) return "N/A";
      if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
      if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
      return n.toFixed(decimals);
    }

    function makeStat(label, value, cls = "") {
      return `<div class="stat"><div class="stat-value ${cls}">${value}</div><div class="stat-label">${label}</div></div>`;
    }

    function drawChart(svgId, data, color) {
      const svg = document.getElementById(svgId);
      if (!svg || data.length < 2) { svg.innerHTML = ''; return; }
      
      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min || 1;
      const w = 400, h = 120, pad = 10;
      
      // Build path
      const points = data.map((v, i) => {
        const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
        const y = h - pad - ((v - min) / range) * (h - 2 * pad);
        return `${x},${y}`;
      });
      
      // Zero line position
      const zeroY = h - pad - ((0 - min) / range) * (h - 2 * pad);
      const showZero = min < 0 && max > 0;
      
      svg.innerHTML = `
        ${showZero ? `<line x1="${pad}" y1="${zeroY}" x2="${w-pad}" y2="${zeroY}" stroke="#333" stroke-dasharray="4"/>` : ''}
        <polyline points="${points.join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>
        <text x="${w-pad}" y="12" fill="#666" font-size="10" text-anchor="end">${fmt(max)}</text>
        <text x="${w-pad}" y="${h-2}" fill="#666" font-size="10" text-anchor="end">${fmt(min)}</text>
      `;
    }

    function renderPerformance(pnl) {
      const sharpe = computeCumulativeSharpe();
      const sharpeStr = sharpe === null ? "N/A" : sharpe.toFixed(2);
      const sharpeClass = sharpe === null ? "" : sharpe > 2 ? "ok" : sharpe > 0 ? "warn" : "err";
      
      const totalReturn = startPnl !== null ? pnl - startPnl : 0;
      const returnClass = totalReturn >= 0 ? "ok" : "err";
      
      document.getElementById("performance-stats").innerHTML = 
        makeStat("PnL", fmt(pnl), pnl >= 0 ? "ok" : "err") +
        makeStat("Session +/-", fmt(totalReturn), returnClass) +
        makeStat("Sharpe", sharpeStr, sharpeClass) +
        makeStat("Samples", allReturns.length.toString());
      
      drawChart("pnl-chart", chartData.pnl, "#4f4");
    }

    function renderExposure(gross, net) {
      const utilization = gross > 0 ? ((gross / 50e6) * 100).toFixed(1) + "%" : "0%";
      const netPct = gross > 0 ? ((Math.abs(net) / gross) * 100).toFixed(0) + "%" : "0%";
      
      document.getElementById("exposure-stats").innerHTML = 
        makeStat("Gross", fmt(gross), gross > 45e6 ? "warn" : "ok") +
        makeStat("Net", fmt(net), Math.abs(net) > 8e6 ? "warn" : "") +
        makeStat("Util", utilization) +
        makeStat("Net/Gross", netPct);
      
      drawChart("exposure-chart", chartData.gross, "#48f");
    }

    function renderPositions(data) {
      const rows = data.positions.map(p => {
        const posClass = p.position !== 0 ? "active-row" : "";
        const pnlClass = (p.unrealized + p.realized) >= 0 ? "ok" : "err";
        return `
        <tr class="${posClass}">
          <td>${p.ticker}</td>
          <td>${p.position.toLocaleString()}</td>
          <td>${p.bid.toFixed(2)}</td>
          <td>${p.ask.toFixed(2)}</td>
          <td>${(p.ask - p.bid).toFixed(3)}</td>
          <td class="${pnlClass}">${fmt(p.unrealized + p.realized)}</td>
        </tr>`;
      }).join("");
      document.getElementById("positions").innerHTML = `
        <table>
          <thead><tr><th>Ticker</th><th>Position</th><th>Bid</th><th>Ask</th><th>Spread</th><th>P&L</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    function renderStrategies(data) {
      // Separate active (non-FLAT) from inactive
      const active = data.strategies.filter(s => s.state !== "FLAT");
      const inactive = data.strategies.filter(s => s.state === "FLAT");
      
      // Active summary at top
      const summaryEl = document.getElementById("active-summary");
      if (active.length > 0) {
        const activeList = active.map(s => `<b>${s.strategy_id}</b> (${s.state}, z=${s.spread?.toFixed(3) || "?"})`).join(" | ");
        summaryEl.innerHTML = `Active: ${activeList}`;
        summaryEl.style.display = "block";
      } else {
        summaryEl.innerHTML = "No active positions";
        summaryEl.style.display = "block";
      }
      
      // Full table
      const renderRow = (s, isActive) => {
        const rowClass = isActive ? "active-row" : "flat-row";
        const stateClass = s.state === "LONG" ? "ok" : s.state === "SHORT" ? "err" : "";
        const strengthClass = s.strength !== null && s.strength > 1.0 ? "warn" : "";
        return `
        <tr class="${rowClass}">
          <td>${s.strategy_id}</td>
          <td class="${stateClass}">${s.state}</td>
          <td>${s.spread === null ? "" : s.spread.toFixed(4)}</td>
          <td class="${strengthClass}">${s.strength === null ? "" : s.strength.toFixed(2)}</td>
          <td>${s.reason}</td>
        </tr>`;
      };
      
      const rows = [...active.map(s => renderRow(s, true)), ...inactive.map(s => renderRow(s, false))].join("");
      document.getElementById("strategies").innerHTML = `
        <table>
          <thead><tr><th>Strategy</th><th>State</th><th>Spread</th><th>Strength</th><th>Reason</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    function renderAllocator(payload) {
      const el = document.getElementById("allocator");
      const a = payload.allocator;
      if (!a) { el.innerHTML = "<div class='card'>Allocator disabled</div>"; return; }

      const diag = a.diagnostics || {};
      const weights = diag.weights || {};
      const edges = diag.edges || {};
      const sigma = diag.sigma_hat || {};
      const netRaw = diag.net_raw || {};
      const ratio = diag.regime_ratio || {};
      const inputs = diag.inputs || {};
      const active = (diag.active || a.active || []).join(", ");

      const names = Array.from(new Set([].concat(Object.keys(weights), Object.keys(edges), Object.keys(sigma), Object.keys(netRaw), Object.keys(ratio))))
        .sort((x, y) => (weights[y] || 0) - (weights[x] || 0));

      const rows = names.map(n => `
        <tr>
          <td>${n}</td>
          <td>${(weights[n] || 0).toFixed(3)}</td>
          <td>${(edges[n] || 0).toFixed(3)}</td>
          <td>${fmt((inputs[n] && inputs[n].s_dollars) || 0, 2)}</td>
          <td>${fmt((inputs[n] && inputs[n].entry_dollars) || 0, 2)}</td>
          <td>${fmt((inputs[n] && inputs[n].rt_cost_dollars) || 0, 4)}</td>
          <td>${fmt(netRaw[n] || 0, 2)}</td>
          <td>${fmt(sigma[n] || 0, 3)}</td>
          <td>${(ratio[n] || 0).toFixed(2)}</td>
        </tr>
      `).join("");

      const reason = diag.reason || "n/a";
      const orderCount = (a.orders || []).length;

      el.innerHTML = `
        <div class="card">
          <div><b>Reason</b>: ${reason} | <b>Active</b>: ${active || "none"} | <b>Orders</b>: ${orderCount}</div>
          <table style="margin-top:8px">
            <thead><tr><th>Book</th><th>w</th><th>edge</th><th>S$</th><th>entry$</th><th>rt_cost$</th><th>net$</th><th>sigma</th><th>sigma/med</th></tr></thead>
            <tbody>${rows || ""}</tbody>
          </table>
        </div>
      `;
    }

    function renderCase(c) {
      const pct = ((c.tick / c.ticks_per_period) * 100).toFixed(0);
      const statusClass = c.status === "ACTIVE" ? "ok" : "warn";
      document.getElementById("case-info").innerHTML = 
        `Period ${c.period}/${c.total_periods} | Tick ${c.tick}/${c.ticks_per_period} (${pct}%) | <span class="${statusClass}">${c.status}</span>`;
    }

    async function refresh() {
      try {
        const [pos, strat] = await Promise.all([fetchJson("/positions"), fetchJson("/strategies")]);
        document.getElementById("status").textContent = "OK";
        
        const pnl = pos.total_pnl;
        const gross = pos.gross_exposure;
        const net = pos.net_exposure;
        
        // Initialize start PnL on first fetch
        if (startPnl === null) startPnl = pnl;
        
        // Track returns for cumulative Sharpe (ALL data, not capped)
        if (chartData.pnl.length > 0) {
          const prevPnl = chartData.pnl[chartData.pnl.length - 1];
          allReturns.push(pnl - prevPnl);
        }
        
        // Chart data (capped for display)
        chartData.pnl.push(pnl);
        chartData.gross.push(gross);
        chartData.net.push(net);
        if (chartData.pnl.length > CHART_POINTS) {
          chartData.pnl.shift();
          chartData.gross.shift();
          chartData.net.shift();
        }
        
        renderCase(strat.case);
        renderPerformance(pnl);
        renderExposure(gross, net);
        renderPositions(pos);
        renderStrategies(strat);
        renderAllocator(strat);
      } catch (e) {
        document.getElementById("status").innerHTML = '<span class="err">ERR: ' + e + '</span>';
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

