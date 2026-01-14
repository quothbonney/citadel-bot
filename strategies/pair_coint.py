import math

from params import PairCointParams
from .base import SpreadStrategy, Order, get_mid


class PairCointStrategy(SpreadStrategy):
    """
    Pair cointegration strategy.

    Trades the spread: z = log(a) - (c + beta * log(b))
    Entry: |z_adj| >= threshold (std or dollar mode)
    Exit: z_adj crosses 0
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
        self._spread_adj: float = 0.0  # For reason formatting

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

    def check_entry_long(self, spread_adj: float) -> bool:
        self._spread_adj = spread_adj
        if self.params.use_std_mode:
            return spread_adj <= -self.params.entry_std * self.params.std
        else:
            dollar_mag = abs(spread_adj) * self._price_a
            return spread_adj < 0 and dollar_mag >= self.params.entry_abs

    def check_entry_short(self, spread_adj: float) -> bool:
        self._spread_adj = spread_adj
        if self.params.use_std_mode:
            return spread_adj >= self.params.entry_std * self.params.std
        else:
            dollar_mag = abs(spread_adj) * self._price_a
            return spread_adj > 0 and dollar_mag >= self.params.entry_abs

    def make_entry_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        a, b = self.params.a, self.params.b
        hb = self._current_hb

        if is_long:
            # z < 0: a undervalued -> buy a, sell b
            return [
                Order(a, 1, 'BUY', self._sec_a.get('ask', self._price_a)),
                Order(b, hb, 'SELL', self._sec_b.get('bid', self._price_b)),
            ]
        else:
            # z > 0: a overvalued -> sell a, buy b
            return [
                Order(a, 1, 'SELL', self._sec_a.get('bid', self._price_a)),
                Order(b, hb, 'BUY', self._sec_b.get('ask', self._price_b)),
            ]

    def make_exit_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        a, b = self.params.a, self.params.b
        hb = self._entry_hb  # Use hedge ratio from entry

        if is_long:
            # Exit long: sell a, buy b
            return [
                Order(a, 1, 'SELL', self._sec_a.get('bid', self._price_a)),
                Order(b, hb, 'BUY', self._sec_b.get('ask', self._price_b)),
            ]
        else:
            # Exit short: buy a, sell b
            return [
                Order(a, 1, 'BUY', self._sec_a.get('ask', self._price_a)),
                Order(b, hb, 'SELL', self._sec_b.get('bid', self._price_b)),
            ]

    def format_entry_reason(self, spread_adj: float) -> str:
        hb = self._current_hb
        if self.params.use_std_mode:
            return f'z={spread_adj:.4f} ({spread_adj/self.params.std:.1f}std) hb={hb:.2f}'
        else:
            dollar_mag = abs(spread_adj) * self._price_a
            return f'z={spread_adj:.4f} (${dollar_mag:.2f}) hb={hb:.2f}'

    def format_hold_reason(self, spread_adj: float) -> str:
        if self.params.use_std_mode:
            return f'z={spread_adj:.4f} ({spread_adj/self.params.std:.1f}std)'
        else:
            dollar_mag = abs(spread_adj) * self._price_a
            return f'z={spread_adj:.4f} (${dollar_mag:.2f})'

    def on_entry(self) -> None:
        self._entry_hb = self._current_hb

    def on_exit(self) -> None:
        self._entry_hb = 0.0
