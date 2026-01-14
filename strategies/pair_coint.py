import math

from params import PairCointParams
from .base import SpreadStrategy, Signal, Order, PositionState, get_mid
from .pyramid import PyramidMixin


class PairCointStrategy(SpreadStrategy, PyramidMixin):
    """
    Pair cointegration strategy with pyramiding support.

    Trades the spread: z = log(a) - (c + beta * log(b))
    Entry: Scale in at each entry_level threshold
    Exit: spread crosses exit_levels, or stop_loss triggered
    """

    def __init__(self, params: PairCointParams) -> None:
        super().__init__(params.strategy_id)
        self.params = params
        self._entry_hb: float = 0.0  # Hedge ratio locked in at entry

        # Cached values from compute_spread (used by make_*_orders)
        self._price_a: float = 0.0
        self._price_b: float = 0.0
        self._sec_a: dict = {}
        self._sec_b: dict = {}
        self._current_hb: float = 0.0
        self._spread_adj: float = 0.0

        # Initialize pyramid state from mixin
        self._init_pyramid(params.pyramid)

    def compute_spread(self, portfolio: dict, case: dict) -> float | None:
        a, b = self.params.a, self.params.b

        self._sec_a = portfolio.get(a, {})
        self._sec_b = portfolio.get(b, {})
        if not self._sec_a or not self._sec_b:
            return None

        self._price_a = get_mid(self._sec_a)
        self._price_b = get_mid(self._sec_b)
        if self._price_a <= 0 or self._price_b <= 0:
            return None

        # Dynamic hedge ratio
        self._current_hb = self.params.beta * (self._price_a / self._price_b)

        # Raw z spread
        return math.log(self._price_a) - (self.params.c + self.params.beta * math.log(self._price_b))

    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        """Full state machine with level-based pyramiding."""
        raw = self.compute_spread(portfolio, case)
        if raw is None:
            return self.hold('missing data')

        spread_adj = self._adjust_for_seasonality(raw, case)
        self._spread_adj = spread_adj

        # Convert z-score to dollar magnitude for threshold comparison
        dollar_mag = abs(spread_adj) * self._price_a
        pyramid = self.params.pyramid

        # --- Risk stops (check first) ---
        if self.state != PositionState.FLAT:
            self._hold_ticks += 1

            # Stop loss (from mixin)
            if self._check_stop_loss(dollar_mag, pyramid):
                return self._exit_position(f'stop_loss hit: ${dollar_mag:.2f}')

        # --- Entry/Scale logic ---
        if self.state == PositionState.FLAT:
            first_level = pyramid.first_entry
            if spread_adj > 0 and dollar_mag >= first_level:
                # z > 0: a overvalued -> short spread
                self.state = PositionState.SHORT
                self._entry_hb = self._current_hb
                size = self._enter_at_level(0, pyramid)
                orders = self._make_orders(is_long=False, quantity=size)
                return self.enter_short(orders, self._format_reason(spread_adj, dollar_mag, size))
            elif spread_adj < 0 and dollar_mag >= first_level:
                # z < 0: a undervalued -> long spread
                self.state = PositionState.LONG
                self._entry_hb = self._current_hb
                size = self._enter_at_level(0, pyramid)
                orders = self._make_orders(is_long=True, quantity=size)
                return self.enter_long(orders, self._format_reason(spread_adj, dollar_mag, size))

        elif self.state == PositionState.LONG:
            # Check scale-up (from mixin)
            scale_result = self._check_scale_up(dollar_mag, pyramid)
            if scale_result is not None:
                lvl, size = scale_result
                orders = self._make_orders(is_long=True, quantity=size)
                return self.scale(orders, f'scale lvl={lvl} size+={size} total={self._current_size}')

            # Check exit levels (spread approaching 0)
            exit_level = self._check_exit(spread_adj, pyramid, is_long=True)
            if exit_level is not None:
                return self._exit_position(f'exit z={spread_adj:.4f}')

        elif self.state == PositionState.SHORT:
            # Check scale-up (from mixin)
            scale_result = self._check_scale_up(dollar_mag, pyramid)
            if scale_result is not None:
                lvl, size = scale_result
                orders = self._make_orders(is_long=False, quantity=size)
                return self.scale(orders, f'scale lvl={lvl} size+={size} total={self._current_size}')

            # Check exit levels (spread approaching 0)
            exit_level = self._check_exit(spread_adj, pyramid, is_long=False)
            if exit_level is not None:
                return self._exit_position(f'exit z={spread_adj:.4f}')

        return self.hold(self._format_hold(spread_adj, dollar_mag))

    def _exit_position(self, reason: str) -> Signal:
        """Exit entire position and reset state."""
        is_long = self.state == PositionState.LONG
        orders = self._make_orders(is_long=not is_long, quantity=self._current_size, use_entry_hb=True)
        self.state = PositionState.FLAT
        self._reset_pyramid(self.params.pyramid)
        self._entry_hb = 0.0
        return self.exit(orders, reason)

    def _make_orders(self, is_long: bool, quantity: float, use_entry_hb: bool = False) -> list[Order]:
        """Generate orders for given direction and quantity."""
        a, b = self.params.a, self.params.b
        hb = self._entry_hb if use_entry_hb else self._current_hb

        if is_long:
            # z < 0: a undervalued -> buy a, sell b
            return [
                Order(a, quantity, 'BUY', self._sec_a.get('ask', self._price_a)),
                Order(b, hb * quantity, 'SELL', self._sec_b.get('bid', self._price_b)),
            ]
        else:
            # z > 0: a overvalued -> sell a, buy b
            return [
                Order(a, quantity, 'SELL', self._sec_a.get('bid', self._price_a)),
                Order(b, hb * quantity, 'BUY', self._sec_b.get('ask', self._price_b)),
            ]

    def _format_reason(self, spread_adj: float, dollar_mag: float, size: int) -> str:
        return f'z={spread_adj:.4f} (${dollar_mag:.2f}) hb={self._current_hb:.2f} size={size}'

    def _format_hold(self, spread_adj: float, dollar_mag: float) -> str:
        return f'z={spread_adj:.4f} (${dollar_mag:.2f}) size={self._current_size}'

    # --- Required abstract methods (for base class compatibility) ---

    def check_entry_long(self, spread_adj: float) -> bool:
        dollar_mag = abs(spread_adj) * self._price_a
        return spread_adj < 0 and dollar_mag >= self.params.pyramid.first_entry

    def check_entry_short(self, spread_adj: float) -> bool:
        dollar_mag = abs(spread_adj) * self._price_a
        return spread_adj > 0 and dollar_mag >= self.params.pyramid.first_entry

    def make_entry_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=is_long, quantity=self.params.pyramid.entry_sizes[0])

    def make_exit_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        return self._make_orders(is_long=not is_long, quantity=self._current_size, use_entry_hb=True)

    def format_entry_reason(self, spread_adj: float) -> str:
        dollar_mag = abs(spread_adj) * self._price_a
        return self._format_reason(spread_adj, dollar_mag, self.params.pyramid.entry_sizes[0])

    def format_hold_reason(self, spread_adj: float) -> str:
        dollar_mag = abs(spread_adj) * self._price_a
        return self._format_hold(spread_adj, dollar_mag)

    def on_entry(self) -> None:
        self._entry_hb = self._current_hb

    def on_exit(self) -> None:
        self._entry_hb = 0.0
