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

from log_config import init_logging
from market import market
from params import StrategyParams, DEFAULT_PARAMS_PATH
from runner import StrategyRunner
from sizer import FixedSizer
from stalker import SessionLoader


@dataclass
class Position:
    """Track position for a single ticker."""
    qty: int = 0
    avg_price: float = 0.0
    realized: float = 0.0

    def fill(self, side: str, fill_qty: int, fill_price: float) -> float:
        """Execute a fill and return realized PnL from this fill."""
        realized_pnl = 0.0

        if side == 'BUY':
            if self.qty >= 0:
                # Adding to long
                total = self.avg_price * self.qty + fill_price * fill_qty
                self.qty += fill_qty
                self.avg_price = total / self.qty if self.qty else 0
            else:
                # Closing short
                close_qty = min(fill_qty, -self.qty)
                realized_pnl = (self.avg_price - fill_price) * close_qty
                self.realized += realized_pnl
                self.qty += fill_qty
                if self.qty > 0:
                    self.avg_price = fill_price
        else:  # SELL
            if self.qty <= 0:
                # Adding to short
                total = self.avg_price * (-self.qty) + fill_price * fill_qty
                self.qty -= fill_qty
                self.avg_price = total / (-self.qty) if self.qty else 0
            else:
                # Closing long
                close_qty = min(fill_qty, self.qty)
                realized_pnl = (fill_price - self.avg_price) * close_qty
                self.realized += realized_pnl
                self.qty -= fill_qty
                if self.qty < 0:
                    self.avg_price = fill_price

        return realized_pnl

    def mtm(self, mid: float) -> float:
        """Mark-to-market unrealized PnL."""
        if self.qty == 0:
            return 0.0
        if self.qty > 0:
            return (mid - self.avg_price) * self.qty
        return (self.avg_price - mid) * (-self.qty)


@dataclass
class StrategyPnL:
    """Track PnL for one strategy."""
    name: str
    positions: dict[str, Position] = field(default_factory=dict)
    trades: int = 0
    costs: float = 0.0
    pnl_curve: list[float] = field(default_factory=list)

    def fill_order(self, ticker: str, side: str, qty: int, price: float, spread: float) -> None:
        if ticker not in self.positions:
            self.positions[ticker] = Position()
        self.positions[ticker].fill(side, qty, price)
        self.costs += spread * qty / 2  # Half-spread cost

    def total_realized(self) -> float:
        return sum(p.realized for p in self.positions.values())

    def total_unrealized(self, prices: dict[str, float]) -> float:
        return sum(p.mtm(prices.get(t, 0)) for t, p in self.positions.items())

    def net_pnl(self, prices: dict[str, float]) -> float:
        return self.total_realized() + self.total_unrealized(prices) - self.costs


class Backtest:
    """Run strategies on historical data."""

    def __init__(self, params: StrategyParams, scale: int = 1) -> None:
        self.scale = scale
        self.sizer = FixedSizer(scale)
        self.runner = StrategyRunner(
            client=None,
            params=params,
            mkt=market,
            sizer=self.sizer,
            dry_run=True,
        )

        # Track PnL per strategy
        self.pnl: dict[str, StrategyPnL] = {}
        for strat in self.runner.strategies:
            self.pnl[strat.strategy_id] = StrategyPnL(strat.strategy_id)

        # Also track allocator if enabled
        if self.runner.allocator:
            self.pnl['allocator'] = StrategyPnL('allocator')
            self._alloc_positions: dict[str, int] = {}  # For injecting into portfolio

    def _get_prices(self, portfolio: dict) -> dict[str, float]:
        prices = {}
        for t in market.all_tickers:
            sec = portfolio.get(t, {})
            bid = sec.get('bid', 0)
            ask = sec.get('ask', 0)
            prices[t] = (bid + ask) / 2 if bid and ask else sec.get('last', 0)
        return prices

    def _get_spread(self, portfolio: dict, ticker: str) -> float:
        sec = portfolio.get(ticker, {})
        return sec.get('ask', 0) - sec.get('bid', 0)

    def process_tick(self, portfolio: dict, case: dict) -> None:
        """Process one tick."""
        # Inject allocator positions if tracking
        if self.runner.allocator and hasattr(self, '_alloc_positions'):
            for t, qty in self._alloc_positions.items():
                if t in portfolio:
                    portfolio[t]['position'] = qty

        # Update allocator PnL (so drawdown-based controls can trigger)
        if self.runner.allocator and 'allocator' in self.pnl:
            prices_now = self._get_prices(portfolio)
            alloc_tracker = self.pnl.get('allocator')
            if alloc_tracker is not None:
                self.runner.allocator.update_pnl(alloc_tracker.net_pnl(prices_now))

        # Run strategies
        signals = self.runner.on_tick(portfolio, case)
        prices = self._get_prices(portfolio)

        for signal in signals:
            sid = signal.strategy_id
            tracker = self.pnl.get(sid)
            if tracker is None:
                continue

            if signal.action in ('enter_long', 'enter_short'):
                tracker.trades += 1

            # Simulate fills
            for order in signal.orders:
                if sid == 'allocator':
                    qty = round(order.quantity)
                else:
                    qty = round(order.quantity * self.scale)

                spread = self._get_spread(portfolio, order.ticker)
                fill_price = order.price

                tracker.fill_order(order.ticker, order.side, qty, fill_price, spread)

                # Update allocator position tracking
                if sid == 'allocator':
                    cur = self._alloc_positions.get(order.ticker, 0)
                    delta = qty if order.side == 'BUY' else -qty
                    self._alloc_positions[order.ticker] = cur + delta

            # Record PnL curve
            tracker.pnl_curve.append(tracker.net_pnl(prices))
        
        # Sync allocator internal state with actual positions
        if self.runner.allocator:
            self.runner.allocator.sync_positions(self._alloc_positions)

    def summary(self) -> None:
        """Print results."""
        print('\n' + '=' * 60)
        print('BACKTEST RESULTS')
        print('=' * 60)

        total = 0.0
        for sid, tracker in self.pnl.items():
            if not tracker.pnl_curve:
                continue

            final_pnl = tracker.pnl_curve[-1]
            total += final_pnl

            print(f'\n{sid}:')
            print(f'  Trades:   {tracker.trades}')
            print(f'  Realized: ${tracker.total_realized():,.2f}')
            print(f'  Costs:    ${tracker.costs:,.2f}')
            print(f'  Net PnL:  ${final_pnl:,.2f}')

            # Compute Sharpe if enough data
            if len(tracker.pnl_curve) > 10:
                returns = np.diff(tracker.pnl_curve)
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390)
                    print(f'  Sharpe:   {sharpe:.2f}')

                # Max drawdown
                curve = np.array(tracker.pnl_curve)
                peak = np.maximum.accumulate(curve)
                dd = (curve - peak).min()
                print(f'  Max DD:   ${dd:,.2f}')

        print(f'\n{"=" * 60}')
        print(f'TOTAL: ${total:,.2f}')
        print('=' * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest strategies')
    parser.add_argument('session', nargs='?', help='Session directory')
    parser.add_argument('--params', '-p', default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument('--scale', '-s', type=int, default=1000)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    init_logging(console_level='DEBUG' if args.verbose else 'WARNING')

    # Find session
    if args.session:
        session_path = Path(args.session)
    else:
        # Try to find most recent session in Downloads
        downloads = Path('/Users/wudd/Downloads')
        sessions = sorted([d for d in downloads.iterdir() if d.is_dir() and 'Citadel' in d.name])
        if not sessions:
            print('No session found')
            sys.exit(1)
        session_path = sessions[-1]

    if not session_path.exists():
        print(f'Session not found: {session_path}')
        sys.exit(1)

    # Load session
    print(f'Loading: {session_path.name}')
    session = SessionLoader(session_path)
    print(f'Tickers: {session.tickers}')

    # Count unique ticks
    tick_count = 0
    for _ in session.ticks():
        tick_count += 1
    print(f'Unique ticks: {tick_count}')

    # Load params
    params = StrategyParams.load(args.params)
    n_strat = len(params.pair_coint) + (1 if params.etf_nav else 0)
    print(f'Strategies: {n_strat}')
    print(f'Scale: {args.scale}x')

    # Run backtest
    bt = Backtest(params, scale=args.scale)

    processed = 0
    for tick in session.ticks():
        portfolio = tick.securities
        case = {'period': tick.period, 'tick': tick.tick, 'status': 'ACTIVE'}

        bt.process_tick(portfolio, case)

        processed += 1
        if processed % 500 == 0:
            print(f'Processed {processed} ticks...')

    print(f'\nProcessed {processed} ticks')
    bt.summary()


if __name__ == '__main__':
    main()
