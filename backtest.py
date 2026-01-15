#!/usr/bin/env python3
"""
Backtest strategies using stalker-recorded session data.

Usage:
    python backtest.py /path/to/session/dir
    python backtest.py  # Uses default session
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Optional empyrical import
try:
    import empyrical as ep
    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False

from log_config import init_logging
from market import market
from params import StrategyParams, DEFAULT_PARAMS_PATH
from runner import StrategyRunner
from sizer import FixedSizer
from stalker import SessionLoader


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""
    strategy_id: str
    n_trades: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


@dataclass
class Position:
    """Track position for PnL calculation."""
    ticker: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


class BacktestRunner:
    """Run strategies on historical data and track simulated PnL."""

    def __init__(self, params: StrategyParams, scale: int = 1) -> None:
        self.params = params
        self.scale = scale
        self.sizer = FixedSizer(scale)
        self.runner = StrategyRunner(
            client=None,  # No live execution
            params=params,
            mkt=market,
            sizer=self.sizer,
            dry_run=True,
        )

        # Track positions per strategy
        self.positions: dict[str, dict[str, Position]] = {}
        for strategy in self.runner.strategies:
            self.positions[strategy.strategy_id] = {}

        # Track results
        self.results: dict[str, BacktestResult] = {}
        for strategy in self.runner.strategies:
            self.results[strategy.strategy_id] = BacktestResult(strategy.strategy_id)

        # PnL curves
        self.pnl_curves: dict[str, list[float]] = {}
        for strategy in self.runner.strategies:
            self.pnl_curves[strategy.strategy_id] = []

        # Add allocator tracking if enabled
        if self.runner.allocator:
            self.positions['allocator'] = {}
            self.results['allocator'] = BacktestResult('allocator')
            self.pnl_curves['allocator'] = []

    def _get_price(self, portfolio: dict, ticker: str, side: str) -> float:
        """Get execution price (bid for sell, ask for buy)."""
        sec = portfolio.get(ticker, {})
        if side == 'BUY':
            return sec.get('ask', sec.get('last', 0))
        return sec.get('bid', sec.get('last', 0))

    def _simulate_fill(self, strategy_id: str, portfolio: dict, order, debug: bool = False) -> float:
        """Simulate order fill and return cost (half-spread)."""
        ticker = order.ticker
        # Allocator already generates absolute sizes; don't scale again
        if strategy_id == 'allocator':
            qty = round(order.quantity)
        else:
            qty = round(order.quantity * self.scale)  # Scale up for strategy orders
        side = order.side

        # Get position
        if ticker not in self.positions[strategy_id]:
            self.positions[strategy_id][ticker] = Position(ticker)
        pos = self.positions[strategy_id][ticker]

        # Get fill price
        price = self._get_price(portfolio, ticker, side)
        if price <= 0:
            return 0.0

        if debug:
            print(f'  FILL: {side} {qty} {ticker} @ {price:.2f}')
            print(f'    pos before: qty={pos.quantity}, avg={pos.avg_price:.2f}, realized={pos.realized_pnl:.2f}')

        # Half-spread cost
        sec = portfolio.get(ticker, {})
        bid = sec.get('bid', 0)
        ask = sec.get('ask', 0)
        spread = (ask - bid) if bid and ask else 0
        cost = (spread / 2) * qty

        # Update position
        if side == 'BUY':
            if pos.quantity >= 0:
                # Adding to long
                total_cost = pos.avg_price * pos.quantity + price * qty
                pos.quantity += qty
                pos.avg_price = total_cost / pos.quantity if pos.quantity else 0
            else:
                # Closing short
                pnl = (pos.avg_price - price) * min(qty, -pos.quantity)
                pos.realized_pnl += pnl
                pos.quantity += qty
                if pos.quantity > 0:
                    pos.avg_price = price
        else:  # SELL
            if pos.quantity <= 0:
                # Adding to short
                total_cost = pos.avg_price * (-pos.quantity) + price * qty
                pos.quantity -= qty
                pos.avg_price = total_cost / (-pos.quantity) if pos.quantity else 0
            else:
                # Closing long
                pnl = (price - pos.avg_price) * min(qty, pos.quantity)
                pos.realized_pnl += pnl
                pos.quantity -= qty
                if pos.quantity < 0:
                    pos.avg_price = price

        if debug:
            print(f'    pos after:  qty={pos.quantity}, avg={pos.avg_price:.2f}, realized={pos.realized_pnl:.2f}')

        return cost

    def _compute_unrealized_pnl(self, strategy_id: str, portfolio: dict) -> float:
        """Compute unrealized PnL for a strategy."""
        unrealized = 0.0
        for ticker, pos in self.positions[strategy_id].items():
            if pos.quantity == 0:
                continue
            sec = portfolio.get(ticker, {})
            bid = sec.get('bid', 0)
            ask = sec.get('ask', 0)
            mid = (bid + ask) / 2 if bid and ask else sec.get('last', 0)
            if pos.quantity > 0:
                unrealized += (mid - pos.avg_price) * pos.quantity
            else:
                unrealized += (pos.avg_price - mid) * (-pos.quantity)
        return unrealized

    def process_tick(self, portfolio: dict, case: dict, tick_num: int = 0, trace_first_n: int = 0) -> list:
        """Process one tick through all strategies."""
        signals = self.runner.on_tick(portfolio, case)

        for signal in signals:
            sid = signal.strategy_id
            result = self.results[sid]

            # Enable debug for first N trades of specified strategy
            debug = (trace_first_n > 0 and
                     signal.action != 'hold' and
                     result.n_trades < trace_first_n)

            if debug:
                print(f'\n=== TICK {tick_num}: {sid} {signal.action} ===')
                print(f'  reason: {signal.reason}')

            if signal.action in ('enter_long', 'enter_short'):
                result.n_trades += 1

            # Simulate fills
            tick_cost = 0.0
            for order in signal.orders:
                cost = self._simulate_fill(sid, portfolio, order, debug=debug)
                tick_cost += cost

            result.costs += tick_cost

            # Compute total PnL
            realized = sum(p.realized_pnl for p in self.positions[sid].values())
            unrealized = self._compute_unrealized_pnl(sid, portfolio)
            result.gross_pnl = realized + unrealized
            result.net_pnl = result.gross_pnl - result.costs

            if debug:
                print(f'  realized={realized:.2f}, unrealized={unrealized:.2f}')
                print(f'  gross={result.gross_pnl:.2f}, costs={result.costs:.2f}, net={result.net_pnl:.2f}')

            self.pnl_curves[sid].append(result.net_pnl)

        return signals

    def _compute_metrics(self, pnl_curve: list[float]) -> dict:
        """Compute risk metrics from PnL curve."""
        if len(pnl_curve) < 2:
            return {}

        pnl = np.array(pnl_curve)
        returns = np.diff(pnl) / (self.scale * 100)  # Normalize by notional

        # Handle edge cases
        if len(returns) == 0 or np.all(returns == 0):
            return {}

        # Compute max drawdown from PnL curve (peak to trough in dollars)
        running_max = np.maximum.accumulate(pnl)
        drawdowns = pnl - running_max
        max_dd = drawdowns.min()

        # Compute Sharpe ratio manually (annualized, assuming 390 ticks per day)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390)
        else:
            sharpe = 0.0

        # Compute Sortino (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252 * 390)
        else:
            sortino = sharpe  # fallback

        result = {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
        }

        # Use empyrical if available for more accurate metrics
        if HAS_EMPYRICAL:
            try:
                result['sharpe'] = ep.sharpe_ratio(returns)
                result['sortino'] = ep.sortino_ratio(returns)
                result['annual_return'] = ep.annual_return(returns)
                result['annual_volatility'] = ep.annual_volatility(returns)
            except Exception:
                pass  # Keep manual calculations

        return result

    def summary(self) -> None:
        """Print backtest summary."""
        print('\n' + '=' * 60)
        print('BACKTEST RESULTS')
        print('=' * 60)

        total_pnl = 0.0
        all_pnl_curves = []

        for sid, result in self.results.items():
            pnl_curve = self.pnl_curves[sid]
            metrics = self._compute_metrics(pnl_curve)

            print(f'\n{sid}:')
            print(f'  Trades:   {result.n_trades}')
            print(f'  Gross:    ${result.gross_pnl:,.2f}')
            print(f'  Costs:    ${result.costs:,.2f}')
            print(f'  Net PnL:  ${result.net_pnl:,.2f}')

            if metrics:
                print(f'  Sharpe:   {metrics["sharpe"]:.2f}')
                print(f'  Sortino:  {metrics["sortino"]:.2f}')
                print(f'  Max DD:   ${metrics["max_drawdown"]:,.2f}')

            total_pnl += result.net_pnl
            all_pnl_curves.append(pnl_curve)

        # Compute aggregate metrics
        if all_pnl_curves:
            min_len = min(len(c) for c in all_pnl_curves)
            combined = np.sum([np.array(c[:min_len]) for c in all_pnl_curves], axis=0)
            agg_metrics = self._compute_metrics(combined.tolist())

            print(f'\n{"=" * 60}')
            print(f'TOTAL NET PnL: ${total_pnl:,.2f}')
            if agg_metrics:
                print(f'Sharpe:        {agg_metrics["sharpe"]:.2f}')
                print(f'Sortino:       {agg_metrics["sortino"]:.2f}')
                print(f'Max Drawdown:  ${agg_metrics["max_drawdown"]:,.2f}')
            print('=' * 60)
        else:
            print(f'\n{"=" * 60}')
            print(f'TOTAL NET PnL: ${total_pnl:,.2f}')
            print('=' * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest strategies on recorded data')
    parser.add_argument('session', nargs='?', help='Path to stalker session directory')
    parser.add_argument('--params', '-p', default=str(DEFAULT_PARAMS_PATH), help='Strategy params file')
    parser.add_argument('--scale', '-s', type=int, default=1000, help='Position size multiplier')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true', help='Print debug info for signals')
    parser.add_argument('--trace', '-t', type=int, default=0, help='Trace first N trades of pair_BBB_DDD')
    args = parser.parse_args()

    init_logging(console_level='DEBUG' if args.verbose else 'WARNING')

    # Default session path - search common locations
    if args.session:
        session_path = Path(args.session)
    else:
        # Try to find a session in common locations
        search_paths = [
            Path.home() / 'Downloads',
            Path.home() / 'Desktop',
            Path.cwd() / 'data',
            Path.cwd(),
        ]
        session_path = None
        for base in search_paths:
            if not base.exists():
                continue
            # Look for directories with 'Citadel' or session-like names
            candidates = sorted([
                d for d in base.iterdir()
                if d.is_dir() and ('Citadel' in d.name or (d / 'meta.json').exists())
            ], key=lambda x: x.stat().st_mtime, reverse=True)
            if candidates:
                session_path = candidates[0]
                break

        if session_path is None:
            print('No session found. Please provide a session path.')
            print('Usage: python backtest.py /path/to/session')
            sys.exit(1)

    if not session_path.exists():
        print(f'Session not found: {session_path}')
        sys.exit(1)

    # Load session
    print(f'Loading session: {session_path.name}')
    session = SessionLoader(session_path)
    print(f'Case: {session.case.get("name")}')
    print(f'Tickers: {session.tickers}')

    # Load params
    params = StrategyParams.load(args.params)
    print(f'Strategies: {len(params.pair_coint)} pair + {"1 ETF-NAV" if params.etf_nav else "0 ETF-NAV"}')
    print(f'Scale: {args.scale}x')

    # Run backtest
    bt = BacktestRunner(params, scale=args.scale)

    # Debug: collect stats for all strategies
    debug_data = {}
    for strat in bt.runner.strategies:
        debug_data[strat.strategy_id] = {'entries': [], 'exits': []}
    if bt.runner.allocator:
        debug_data['allocator'] = {'entries': [], 'exits': []}

    tick_count = 0
    for tick in session.ticks():
        portfolio = tick.securities
        case = {'period': tick.period, 'tick': tick.tick, 'status': 'ACTIVE'}

        # Inject tracked positions into portfolio for allocator
        if bt.runner.allocator and 'allocator' in bt.positions:
            for ticker, pos in bt.positions['allocator'].items():
                if ticker in portfolio:
                    portfolio[ticker]['position'] = pos.quantity

        signals = bt.process_tick(portfolio, case, tick_num=tick_count, trace_first_n=args.trace)

        # Track entries/exits
        if args.debug:
            for sig in signals:
                if sig.action in ('enter_long', 'enter_short'):
                    debug_data.get(sig.strategy_id, {}).get('entries', []).append((tick_count, sig.action, sig.reason))
                elif sig.action == 'exit':
                    debug_data.get(sig.strategy_id, {}).get('exits', []).append((tick_count, sig.reason))

        tick_count += 1
        if tick_count % 500 == 0:
            print(f'Processed {tick_count} ticks...')

    print(f'\nProcessed {tick_count} total ticks')
    bt.summary()

    # Print debug stats
    if args.debug:
        print('\n' + '=' * 60)
        print('DEBUG: Trade Activity')
        print('=' * 60)

        for sid, data in debug_data.items():
            entries = data.get('entries', [])
            exits = data.get('exits', [])
            if not entries and not exits:
                continue
            print(f'\n{sid}:')
            print(f'  entries:  {len(entries)}')
            print(f'  exits:    {len(exits)}')

            # Show first few entries
            if entries:
                print(f'  first 5 entries:')
                for t, action, reason in entries[:5]:
                    print(f'    tick {t}: {action} - {reason}')


if __name__ == '__main__':
    main()
