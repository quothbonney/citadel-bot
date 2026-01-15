import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from allocator import PortfolioAllocator, AllocatorConfig
from market import Market, market
from params import StrategyParams
from sizer import Sizer, UnitSizer
from strategies import SignalStrategy, Signal, PairCointStrategy, EtfNavStrategy
from strategies.base import get_mid


@dataclass
class StrategyPnL:
    """Track PnL for a strategy based on its allocation weight."""
    name: str
    pnl: float = 0.0
    pnl_history: list = field(default_factory=list)
    active_ticks: int = 0

    @property
    def sharpe(self) -> float:
        """Compute Sharpe ratio from PnL history."""
        if len(self.pnl_history) < 10:
            return 0.0
        returns = np.diff(self.pnl_history)
        if len(returns) == 0 or np.std(returns) < 1e-9:
            return 0.0
        # Annualize: ~780 ticks per period, 2 periods per session
        return np.mean(returns) / np.std(returns) * np.sqrt(780)


class PnLTracker:
    """Track PnL attribution across strategies."""

    def __init__(self, strategy_names: list[str]):
        self.strategies = {name: StrategyPnL(name) for name in strategy_names}
        self.strategies['total'] = StrategyPnL('total')
        self._prev_pnl = 0.0
        self._strategy_weights: dict[str, float] = {}

    def update(self, total_pnl: float, active_strategies: list[str],
               weights: dict[str, float] | None = None) -> None:
        """Update PnL tracking with new total PnL and active strategies.

        Args:
            total_pnl: Current total portfolio PnL
            active_strategies: List of currently active strategy names
            weights: Optional dict of strategy name -> weight (0-1)
        """
        # Compute PnL delta
        delta = total_pnl - self._prev_pnl
        self._prev_pnl = total_pnl

        # Update total
        self.strategies['total'].pnl = total_pnl
        self.strategies['total'].pnl_history.append(total_pnl)

        # Store weights for attribution
        if weights:
            self._strategy_weights = weights

        # Attribute delta to active strategies proportionally
        if active_strategies and delta != 0:
            # If we have weights, use them; otherwise equal weight
            if self._strategy_weights:
                total_weight = sum(self._strategy_weights.get(s, 0) for s in active_strategies)
                for name in active_strategies:
                    if name in self.strategies:
                        w = self._strategy_weights.get(name, 0) / total_weight if total_weight > 0 else 0
                        self.strategies[name].pnl += delta * w
                        self.strategies[name].active_ticks += 1
            else:
                share = delta / len(active_strategies)
                for name in active_strategies:
                    if name in self.strategies:
                        self.strategies[name].pnl += share
                        self.strategies[name].active_ticks += 1

        # Update history for all strategies
        for name, spnl in self.strategies.items():
            if name != 'total':
                spnl.pnl_history.append(spnl.pnl)

    def get_stats(self) -> dict:
        """Get current stats for all strategies."""
        return {
            name: {
                'pnl': spnl.pnl,
                'sharpe': spnl.sharpe,
                'active_ticks': spnl.active_ticks,
            }
            for name, spnl in self.strategies.items()
        }


class StrategyRunner:
    """Orchestrates multiple independent strategies."""

    def __init__(
        self,
        client: RotmanInteractiveTraderApi | None,
        params: StrategyParams,
        mkt: Market = market,
        sizer: Sizer | None = None,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.params = params
        self.market = mkt
        self.sizer = sizer or UnitSizer()
        self.dry_run = dry_run

        self.strategies: list[SignalStrategy] = []
        self._build_strategies()

        # Initialize PnL tracker
        strategy_names = [s.strategy_id for s in self.strategies]
        self.pnl_tracker = PnLTracker(strategy_names)
        self._last_active: list[str] = []
        self._last_weights: dict[str, float] = {}

        # Initialize allocator if enabled
        self.allocator: PortfolioAllocator | None = None
        if params.allocator and params.allocator.enabled:
            config = AllocatorConfig(
                gross_limit=params.allocator.gross_limit,
                net_limit=params.allocator.net_limit,
                max_shares=params.allocator.max_shares or mkt.max_shares,
                turnover_k=params.allocator.turnover_k,
                min_threshold=params.allocator.min_threshold,
                top_n=params.allocator.top_n,
                stop_loss_mult=params.allocator.stop_loss_mult,
                take_profit_mult=params.allocator.take_profit_mult,
                max_hold_ticks=params.allocator.max_hold_ticks,
            )
            self.allocator = PortfolioAllocator(config, params.width)
            logging.info('Allocator enabled: gross=$%.0fM min_thresh=%.2f top_n=%d SL=%.1fx TP=%.1fx max_hold=%d',
                         config.gross_limit / 1e6, config.min_threshold, config.top_n,
                         config.stop_loss_mult, config.take_profit_mult, config.max_hold_ticks)

    def update_pnl(self, total_pnl: float) -> None:
        """Update PnL tracking with current total PnL."""
        self.pnl_tracker.update(total_pnl, self._last_active, self._last_weights)

    def get_pnl_stats(self) -> dict:
        """Get PnL stats for all strategies."""
        return self.pnl_tracker.get_stats()

    def _build_strategies(self) -> None:
        """Instantiate all enabled strategies from params."""
        # Pair cointegration strategies
        for p in self.params.pair_coint:
            if p.enabled:
                self.strategies.append(PairCointStrategy(p))
                logging.info('Loaded strategy: %s', p.strategy_id)

        # ETF-NAV strategy
        if self.params.etf_nav and self.params.etf_nav.enabled:
            self.strategies.append(EtfNavStrategy(self.params.etf_nav, self.market))
            logging.info('Loaded strategy: %s', self.params.etf_nav.strategy_id)

        logging.info('Total strategies: %d', len(self.strategies))

    def _check_risk(self, portfolio: dict, signal: Signal) -> bool:
        """Check if executing signal would exceed risk limits."""
        # Extract current positions and prices
        positions = {}
        prices = {}
        for ticker in self.market.all_tickers:
            sec = portfolio.get(ticker, {})
            positions[ticker] = sec.get('position', 0)
            bid = sec.get('bid', 0)
            ask = sec.get('ask', 0)
            prices[ticker] = (bid + ask) / 2 if bid and ask else sec.get('last', 0)

        # Project positions after trade
        for order in signal.orders:
            qty = self.sizer.scale(order.quantity, portfolio, self.market)
            delta = qty if order.side == 'BUY' else -qty
            positions[order.ticker] = positions.get(order.ticker, 0) + delta

        # Check limits
        ok, gross, net = self.market.check_limits(positions, prices)
        if not ok:
            logging.warning(
                'Risk limit exceeded for %s: gross=%.0f net=%.0f',
                signal.strategy_id, gross, net
            )
        return ok

    def _execute(self, portfolio: dict, signal: Signal) -> None:
        """Execute a signal by placing orders."""
        if not signal.orders:
            return

        if not self._check_risk(portfolio, signal):
            return

        for order in signal.orders:
            qty = self.sizer.scale(order.quantity, portfolio, self.market)
            action = OrderAction.BUY if order.side == 'BUY' else OrderAction.SELL

            if self.dry_run or self.client is None:
                logging.info(
                    '[DRY RUN] %s: %s %d %s @ %.2f',
                    signal.strategy_id, order.side, qty, order.ticker, order.price
                )
            else:
                try:
                    resp = self.client.place_order(
                        order.ticker,
                        OrderType.LIMIT,
                        qty,
                        action,
                        price=order.price,
                    )
                    logging.info(
                        '%s: %s %d %s @ %.2f -> order_id=%s',
                        signal.strategy_id, order.side, qty, order.ticker, order.price,
                        resp.get('order_id') if isinstance(resp, dict) else resp
                    )
                except Exception as e:
                    logging.error('%s: order failed: %s', signal.strategy_id, e)

    def on_tick(self, portfolio: dict, case: dict) -> list[Signal]:
        """Process one tick across all strategies."""
        if self.allocator:
            return self._on_tick_allocated(portfolio, case)
        return self._on_tick_independent(portfolio, case)

    def _on_tick_independent(self, portfolio: dict, case: dict) -> list[Signal]:
        """Process tick with independent strategy execution (original behavior)."""
        signals = []

        for strategy in self.strategies:
            signal = strategy.compute_signal(portfolio, case)
            signals.append(signal)

            if signal.action != 'hold':
                logging.info(
                    '%s: %s (%s)',
                    signal.strategy_id, signal.action, signal.reason
                )
                self._execute(portfolio, signal)

        return signals

    def _on_tick_allocated(self, portfolio: dict, case: dict) -> list[Signal]:
        """Process tick with top-N weighted allocation."""
        # Collect signal specs from ALL strategies (allocator decides which to use)
        specs = []
        for strategy in self.strategies:
            spec = strategy.get_signal_spec(portfolio, case)
            if spec is not None:
                specs.append(spec)

        # Get prices
        prices = {t: get_mid(portfolio.get(t, {})) for t in self.market.all_tickers}

        # Get current positions
        current_pos = {t: portfolio.get(t, {}).get('position', 0) for t in self.market.all_tickers}

        # Debug: log signal strengths
        if specs:
            above_thresh = [s for s in specs if s.abs_signal >= self.allocator.config.min_threshold]
            logging.debug('Signals: %d total, %d above threshold (%.2f)',
                         len(specs), len(above_thresh), self.allocator.config.min_threshold)
            for s in sorted(specs, key=lambda x: x.strength, reverse=True)[:4]:
                logging.debug('  %s: signal=%.4f sigma=%.4f strength=%.2f %s',
                             s.name, s.signal, s.sigma, s.strength,
                             '*' if s.abs_signal >= self.allocator.config.min_threshold else '')

        # Allocate (returns target positions and list of active strategy names)
        target_pos, active_names = self.allocator.allocate(specs, prices, current_pos)

        # Track active strategies and weights for PnL attribution
        self._last_active = active_names
        if active_names:
            # Compute weights from spec strengths
            active_specs = [s for s in specs if s.name in active_names]
            total_strength = sum(s.strength for s in active_specs)
            if total_strength > 0:
                self._last_weights = {s.name: s.strength / total_strength for s in active_specs}
        else:
            self._last_weights = {}

        # Convert to orders (with debug logging)
        orders = self.allocator.positions_to_orders(target_pos, current_pos, prices, debug=True)

        # Execute orders
        if orders:
            reason = f'active: {", ".join(active_names)}' if active_names else 'flattening'
            alloc_signal = Signal('allocator', 'trade', orders, reason=reason)
            for order in orders:
                self._execute_allocator_order(portfolio, order)
            logging.info('Allocator: %d orders | %s', len(orders), reason)
            return [alloc_signal]

        reason = f'active: {", ".join(active_names)}' if active_names else 'no signals above threshold'
        return [Signal('allocator', 'hold', reason=reason)]

    def _execute_allocator_order(self, portfolio: dict, order) -> None:
        """Execute a single allocator order."""
        action = OrderAction.BUY if order.side == 'BUY' else OrderAction.SELL
        qty = int(order.quantity)

        if self.dry_run or self.client is None:
            logging.info('[DRY RUN] allocator: %s %d %s @ %.2f',
                         order.side, qty, order.ticker, order.price)
        else:
            try:
                resp = self.client.place_order(
                    order.ticker,
                    OrderType.LIMIT,
                    qty,
                    action,
                    price=order.price,
                )
                logging.info('allocator: %s %d %s @ %.2f -> order_id=%s',
                             order.side, qty, order.ticker, order.price,
                             resp.get('order_id') if isinstance(resp, dict) else resp)
            except Exception as e:
                logging.error('allocator: order failed: %s', e)
