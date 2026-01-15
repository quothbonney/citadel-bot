import logging
from dataclasses import dataclass
from typing import Protocol

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from allocator import PortfolioAllocator, AllocatorConfig
from market import Market, market
from params import StrategyParams
from sizer import Sizer, UnitSizer
from strategies import SignalStrategy, Signal, PairCointStrategy, EtfNavStrategy
from strategies.base import get_mid


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
            )
            self.allocator = PortfolioAllocator(config, params.width)
            logging.info('Allocator enabled: gross=$%.0fM net=$%.0fM min_thresh=%.2f top_n=%d',
                         config.gross_limit / 1e6, config.net_limit / 1e6,
                         config.min_threshold, config.top_n)

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

        # Allocate (returns target positions and list of active strategy names)
        target_pos, active_names = self.allocator.allocate(specs, prices, current_pos)

        # Convert to orders
        orders = self.allocator.positions_to_orders(target_pos, current_pos, prices)

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
