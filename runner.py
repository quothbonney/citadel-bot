import logging
from dataclasses import dataclass
from typing import Protocol

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from allocator import Allocator, AllocatorConfig, Signal as AllocSignal
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
        self.allocator: Allocator | None = None
        if params.allocator and params.allocator.enabled:
            config = AllocatorConfig(
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
            )
            self.allocator = Allocator(config)
            logging.info('Allocator enabled: gross=$%.0fM net=$%.0fM top_n=%d horizon=%d lambda=%.2f turnover=%.0f%%',
                         config.gross_limit / 1e6, config.net_limit / 1e6,
                         config.top_n, config.horizon_bars, config.switch_lambda, config.turnover_pct * 100)

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
        # Collect signals from all strategies
        signals = []
        for strategy in self.strategies:
            sig = self._get_alloc_signal(strategy, portfolio, case)
            if sig is not None:
                signals.append(sig)

        # Get prices
        prices = {t: get_mid(portfolio.get(t, {})) for t in self.market.all_tickers}

        # Get current positions
        current_pos = {t: portfolio.get(t, {}).get('position', 0) for t in self.market.all_tickers}

        # Allocate (returns target positions and list of active strategy names)
        target_pos, active_names = self.allocator.allocate(signals, prices, current_pos)

        # Convert to orders
        # NOTE: turnover cap already limits per-tick position changes; don't add a huge deadband here.
        orders = self.allocator.to_orders(target_pos, current_pos, prices, min_delta=1)

        # Execute orders
        if orders:
            reason = f'active: {", ".join(active_names)}' if active_names else 'flattening'
            alloc_signal = Signal('allocator', 'trade', orders, reason=reason)
            for order in orders:
                self._execute_allocator_order(portfolio, order)
            logging.info('Allocator: %d orders | %s', len(orders), reason)
            # Sync allocator state with actual positions (for next tick)
            self.allocator.sync_positions(current_pos)
            return [alloc_signal]

        reason = f'active: {", ".join(active_names)}' if active_names else 'no signals above threshold'
        # Even on hold, sync positions
        self.allocator.sync_positions(current_pos)
        return [Signal('allocator', 'hold', reason=reason)]

    def _get_alloc_signal(self, strategy: SignalStrategy, portfolio: dict, case: dict) -> AllocSignal | None:
        """Extract allocator signal from strategy."""
        # Run compute_signal to get the spread_adj
        strategy.compute_signal(portfolio, case)
        spread = getattr(strategy, '_spread_adj', None)
        if spread is None:
            return None

        width = self.params.width or {}

        # Build legs dict based on strategy type
        if isinstance(strategy, PairCointStrategy):
            p = strategy.params
            prices = {t: get_mid(portfolio.get(t, {})) for t in self.market.all_tickers}
            pa = prices.get(p.a, 1)
            pb = prices.get(p.b, 1)
            hb = p.beta * (pa / pb) if pb > 0 else 1.0
            # legs for +1 unit SHORT spread: sell a, buy b
            legs = {p.a: -1.0, p.b: hb}
            if p.std <= 0:
                raise ValueError(f"invalid std for {p.strategy_id}: {p.std}")
            z = spread / p.std
            s_dollars = float(z) * float(pa)
            entry_dollars = float(p.pyramid.first_entry) * float(pa)
            rt_cost_dollars = abs(legs[p.a]) * float(width.get(p.a, 0.0)) + abs(legs[p.b]) * float(width.get(p.b, 0.0))
            return AllocSignal(
                name=strategy.strategy_id,
                s_dollars=s_dollars,
                entry_dollars=entry_dollars,
                rt_cost_dollars=rt_cost_dollars,
                legs=legs,
            )

        elif isinstance(strategy, EtfNavStrategy):
            # legs for +1 unit SHORT spread: sell ETF, buy stocks
            legs = {'ETF': -1.0, 'AAA': 0.25, 'BBB': 0.25, 'CCC': 0.25, 'DDD': 0.25}
            rt_cost_dollars = (
                abs(legs['ETF']) * float(width.get('ETF', 0.0))
                + abs(legs['AAA']) * float(width.get('AAA', 0.0))
                + abs(legs['BBB']) * float(width.get('BBB', 0.0))
                + abs(legs['CCC']) * float(width.get('CCC', 0.0))
                + abs(legs['DDD']) * float(width.get('DDD', 0.0))
            )
            return AllocSignal(
                name=strategy.strategy_id,
                s_dollars=spread,
                entry_dollars=strategy.params.pyramid.first_entry,
                rt_cost_dollars=rt_cost_dollars,
                legs=legs,
            )

        return None

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
