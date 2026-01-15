from __future__ import annotations

from typing import TYPE_CHECKING

from params import EtfNavParams
from market import Market, market
from .base import SpreadStrategy, Signal, Order, PositionState, get_mid
from .pyramid import PyramidMixin

if TYPE_CHECKING:
    from allocator import StrategySpec


class EtfNavStrategy(SpreadStrategy, PyramidMixin):
    """
    ETF-NAV arbitrage strategy with pyramiding and risk management.

    Trades the spread: ETF - mean(AAA, BBB, CCC, DDD)
    Entry: Scale in at each entry_level threshold
    Exit: spread crosses exit_levels, or stop_loss triggered
    """

    def __init__(self, params: EtfNavParams, mkt: Market = market) -> None:
        super().__init__(params.strategy_id)
        self.params = params
        self.market = mkt

        # Cached values from compute_spread (used by make_*_orders)
        self._sec_etf: dict = {}
        self._stock_secs: dict[str, dict] = {}
        self._etf_mid: float = 0.0
        self._stock_mids: dict[str, float] = {}

        # Initialize pyramid state from mixin
        self._init_pyramid(params.pyramid)
        self._spread_adj: float = 0.0

    def compute_spread(self, portfolio: dict, case: dict) -> float | None:
        etf = self.market.etf
        stocks = self.market.stocks

        self._sec_etf = portfolio.get(etf, {})
        if not self._sec_etf:
            return None

        self._stock_secs = {s: portfolio.get(s, {}) for s in stocks}
        if not all(self._stock_secs.values()):
            return None

        self._etf_mid = get_mid(self._sec_etf)
        self._stock_mids = {s: get_mid(self._stock_secs[s]) for s in stocks}
        if self._etf_mid <= 0 or any(p <= 0 for p in self._stock_mids.values()):
            return None

        # NAV = average of stock prices
        nav = sum(self._stock_mids.values()) / len(stocks)
        return self._etf_mid - nav

    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        """Full state machine with level-based pyramiding and risk management."""
        raw = self.compute_spread(portfolio, case)
        if raw is None:
            return self.hold('missing data')

        spread_adj = self._adjust_for_seasonality(raw, case)
        self._spread_adj = spread_adj
        tick = case.get('tick', 0)
        abs_spread = abs(spread_adj)
        pyramid = self.params.pyramid

        # --- Risk stops (check first, before any other logic) ---
        if self.state != PositionState.FLAT:
            self._hold_ticks += 1

            # Stop loss (from mixin)
            if self._check_stop_loss(abs_spread, pyramid):
                return self._exit_position(f'stop_loss hit: {spread_adj:.4f}')

            # EOD flat (ETF-NAV specific)
            if self.params.eod_flat and tick >= 390:
                return self._exit_position(f'eod_flat: tick={tick}')

        # --- Entry/Scale logic ---
        if self.state == PositionState.FLAT:
            first_level = pyramid.first_entry
            if spread_adj >= first_level:
                # ETF overvalued: short spread
                self.state = PositionState.SHORT
                size = self._enter_at_level(0, pyramid)
                orders = self._make_orders(is_long=False, quantity=size)
                return self.enter_short(orders, f'spread_adj={spread_adj:.4f} lvl=0 size={size}')
            elif spread_adj <= -first_level:
                # ETF undervalued: long spread
                self.state = PositionState.LONG
                size = self._enter_at_level(0, pyramid)
                orders = self._make_orders(is_long=True, quantity=size)
                return self.enter_long(orders, f'spread_adj={spread_adj:.4f} lvl=0 size={size}')

        elif self.state == PositionState.LONG:
            # Check scale-up (from mixin)
            scale_result = self._check_scale_up(abs_spread, pyramid)
            if scale_result is not None:
                lvl, size = scale_result
                orders = self._make_orders(is_long=True, quantity=size)
                return self.scale(orders, f'scale lvl={lvl} size+={size} total={self._current_size}')

            # Check exit levels (from mixin)
            exit_level = self._check_exit(spread_adj, pyramid, is_long=True)
            if exit_level is not None:
                return self._exit_position(f'exit_level {exit_level}: spread={spread_adj:.4f}')

        elif self.state == PositionState.SHORT:
            # Check scale-up (from mixin)
            scale_result = self._check_scale_up(abs_spread, pyramid)
            if scale_result is not None:
                lvl, size = scale_result
                orders = self._make_orders(is_long=False, quantity=size)
                return self.scale(orders, f'scale lvl={lvl} size+={size} total={self._current_size}')

            # Check exit levels (from mixin)
            exit_level = self._check_exit(spread_adj, pyramid, is_long=False)
            if exit_level is not None:
                return self._exit_position(f'exit_level {exit_level}: spread={spread_adj:.4f}')

        return self.hold(f'spread_adj={spread_adj:.4f} size={self._current_size}')

    def _exit_position(self, reason: str) -> Signal:
        """Exit entire position and reset state."""
        is_long = self.state == PositionState.LONG
        orders = self._make_orders(is_long=not is_long, quantity=self._current_size)
        self.state = PositionState.FLAT
        self._reset_pyramid(self.params.pyramid)
        return self.exit(orders, reason)

    def _make_orders(self, is_long: bool, quantity: float) -> list[Order]:
        """Generate orders for given direction and quantity.

        Hedge ratio: 1 ETF : 0.25 of each stock (since NAV = avg of 4 stocks)
        """
        etf = self.market.etf
        stocks = self.market.stocks
        stock_qty = quantity * 0.25  # Each stock is 1/4 of NAV

        if is_long:
            # Long spread: buy ETF, sell stocks
            orders = [Order(etf, quantity, 'BUY', self._sec_etf.get('ask', self._etf_mid))]
            for s in stocks:
                orders.append(Order(s, stock_qty, 'SELL', self._stock_secs[s].get('bid', self._stock_mids[s])))
        else:
            # Short spread: sell ETF, buy stocks
            orders = [Order(etf, quantity, 'SELL', self._sec_etf.get('bid', self._etf_mid))]
            for s in stocks:
                orders.append(Order(s, stock_qty, 'BUY', self._stock_secs[s].get('ask', self._stock_mids[s])))
        return orders

    # --- Required abstract methods (delegated to _make_orders) ---

    def check_entry_long(self, spread_adj: float) -> bool:
        return spread_adj <= -self.params.pyramid.first_entry

    def check_entry_short(self, spread_adj: float) -> bool:
        return spread_adj >= self.params.pyramid.first_entry

    def make_entry_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=is_long, quantity=self.params.pyramid.entry_sizes[0])

    def make_exit_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=not is_long, quantity=self._current_size)

    def format_entry_reason(self, spread_adj: float) -> str:
        return f'spread_adj={spread_adj:.4f}'

    def format_hold_reason(self, spread_adj: float) -> str:
        return f'spread_adj={spread_adj:.4f} size={self._current_size}'

    # --- Allocator integration ---

    def get_signal_spec(self, portfolio: dict, case: dict) -> StrategySpec | None:
        """Return allocator input (always, for top-N ranking)."""
        raw = self.compute_spread(portfolio, case)
        if raw is None:
            return None

        spread_adj = self._adjust_for_seasonality(raw, case)

        from allocator import StrategySpec
        return StrategySpec(
            name='ETF-NAV',
            signal=spread_adj,  # + = ETF overvalued = short spread (in dollars)
            sigma=self.params.std,  # Volatility estimate for consistent ranking
            build_pos=self._build_pos_per_unit,
        )

    def _build_pos_per_unit(self, prices: dict) -> dict:
        """Position for +1 unit of short spread (sell ETF, buy stocks)."""
        return {
            'ETF': -1.0,
            'AAA': 0.25,
            'BBB': 0.25,
            'CCC': 0.25,
            'DDD': 0.25,
            'IND': 0.0,
        }
