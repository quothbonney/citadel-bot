from params import EtfNavParams
from market import Market, market
from .base import SpreadStrategy, Signal, Order, PositionState, get_mid


class EtfNavStrategy(SpreadStrategy):
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

        # Pyramid state
        self._triggered_levels: list[bool] = [False] * len(params.entry_levels)
        self._current_size: int = 0  # Total units currently held
        self._hold_ticks: int = 0
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

        # --- Risk stops (check first, before any other logic) ---
        if self.state != PositionState.FLAT:
            self._hold_ticks += 1

            # Stop loss
            if self.params.stop_loss is not None and abs_spread >= self.params.stop_loss:
                return self._exit_position(f'stop_loss hit: {spread_adj:.4f}')

            # EOD flat
            if self.params.eod_flat and tick >= 390:
                return self._exit_position(f'eod_flat: tick={tick}')

        # --- Entry/Scale logic ---
        if self.state == PositionState.FLAT:
            # Check first entry level
            first_level = self.params.entry_levels[0]
            if spread_adj >= first_level:
                # ETF overvalued: short spread
                self.state = PositionState.SHORT
                return self._enter_at_level(0, is_long=False, spread_adj=spread_adj)
            elif spread_adj <= -first_level:
                # ETF undervalued: long spread
                self.state = PositionState.LONG
                return self._enter_at_level(0, is_long=True, spread_adj=spread_adj)

        elif self.state == PositionState.LONG:
            # Check scale-up (spread getting more negative)
            scale_signal = self._check_scale_up(abs_spread, is_long=True)
            if scale_signal is not None:
                return scale_signal

            # Check exit levels (spread approaching 0)
            exit_signal = self._check_exit(spread_adj, is_long=True)
            if exit_signal is not None:
                return exit_signal

        elif self.state == PositionState.SHORT:
            # Check scale-up (spread getting more positive)
            scale_signal = self._check_scale_up(abs_spread, is_long=False)
            if scale_signal is not None:
                return scale_signal

            # Check exit levels (spread approaching 0)
            exit_signal = self._check_exit(spread_adj, is_long=False)
            if exit_signal is not None:
                return exit_signal

        return self.hold(f'spread_adj={spread_adj:.4f} size={self._current_size}')

    def _enter_at_level(self, level_idx: int, is_long: bool, spread_adj: float) -> Signal:
        """Enter position at a specific level."""
        size = self.params.entry_sizes[level_idx]
        self._triggered_levels[level_idx] = True
        self._current_size = size
        self._hold_ticks = 0
        orders = self._make_orders(is_long=is_long, quantity=size)
        return self.enter_long(orders, f'spread_adj={spread_adj:.4f} lvl=0 size={size}') if is_long \
            else self.enter_short(orders, f'spread_adj={spread_adj:.4f} lvl=0 size={size}')

    def _check_scale_up(self, abs_spread: float, is_long: bool) -> Signal | None:
        """Check if we should scale up at the next entry level."""
        for i, (level, size) in enumerate(zip(self.params.entry_levels, self.params.entry_sizes)):
            if not self._triggered_levels[i] and abs_spread >= level:
                self._triggered_levels[i] = True
                self._current_size += size
                orders = self._make_orders(is_long=is_long, quantity=size)
                return self.scale(orders, f'scale lvl={i} size+={size} total={self._current_size}')
        return None

    def _check_exit(self, spread_adj: float, is_long: bool) -> Signal | None:
        """Check if spread has crossed an exit level."""
        # For long: spread was negative, exit when it crosses towards 0
        # For short: spread was positive, exit when it crosses towards 0
        for exit_level in self.params.exit_levels:
            if is_long and spread_adj >= exit_level:
                return self._exit_position(f'exit_level {exit_level}: spread={spread_adj:.4f}')
            elif not is_long and spread_adj <= exit_level:
                return self._exit_position(f'exit_level {exit_level}: spread={spread_adj:.4f}')
        return None

    def _exit_position(self, reason: str) -> Signal:
        """Exit entire position and reset state."""
        is_long = self.state == PositionState.LONG
        orders = self._make_orders(is_long=not is_long, quantity=self._current_size)
        self.state = PositionState.FLAT
        self._triggered_levels = [False] * len(self.params.entry_levels)
        self._current_size = 0
        self._hold_ticks = 0
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
        return spread_adj <= -self.params.entry_levels[0]

    def check_entry_short(self, spread_adj: float) -> bool:
        return spread_adj >= self.params.entry_levels[0]

    def make_entry_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=is_long, quantity=self.params.entry_sizes[0])

    def make_exit_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=not is_long, quantity=self._current_size)

    def format_entry_reason(self, spread_adj: float) -> str:
        return f'spread_adj={spread_adj:.4f}'

    def format_hold_reason(self, spread_adj: float) -> str:
        return f'spread_adj={spread_adj:.4f} size={self._current_size}'
