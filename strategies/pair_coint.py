import math
from collections import defaultdict

from params import PairCointParams
from .base import SignalStrategy, Signal, Order, PositionState


class RunningMean:
    """Online computation of expanding mean."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n

    def get(self) -> float | None:
        return self.mean if self.n > 0 else None


class PairCointStrategy(SignalStrategy):
    """
    Pair cointegration strategy.

    Trades the spread: z = log(a) - (c + beta * log(b))
    Entry: |z_adj| >= entry_std * std (z is N std devs from mean)
    Exit: z_adj crosses 0
    """

    def __init__(self, params: PairCointParams) -> None:
        super().__init__(params.strategy_id)
        self.params = params

        # Seasonality: expanding mean of z per timestep
        self._z_means: dict[int, RunningMean] = defaultdict(RunningMean)
        self._z_adj_cache: dict[tuple[int, int], float] = {}  # Cache z_adj per (period, tick)
        self._last_z_adj: float | None = None
        self._entry_hb: float = 0.0  # Track hedge ratio from entry (raw, not rounded)

    def _get_mid(self, sec: dict) -> float:
        bid = sec.get('bid', 0)
        ask = sec.get('ask', 0)
        if bid and ask:
            return (bid + ask) / 2
        return sec.get('last', 0)

    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        a, b = self.params.a, self.params.b

        sec_a = portfolio.get(a)
        sec_b = portfolio.get(b)
        if not sec_a or not sec_b:
            return self.hold('missing securities')

        price_a = self._get_mid(sec_a)
        price_b = self._get_mid(sec_b)
        if price_a <= 0 or price_b <= 0:
            return self.hold('invalid prices')

        # Compute raw z
        z = math.log(price_a) - (self.params.c + self.params.beta * math.log(price_b))

        # Seasonality adjustment
        period = case.get('period', 0)
        tick = case.get('tick', 0)
        tick_key = (period, tick)
        z_mean = self._z_means[tick]

        # Only compute z_adj once per (period, tick) to handle multiple snapshots per tick
        if tick_key in self._z_adj_cache:
            # Already processed this tick: use cached value
            z_adj = self._z_adj_cache[tick_key]
        else:
            # First time seeing this tick in this period: use previous periods' mean
            prev_mean = z_mean.get()
            z_adj = z - prev_mean if prev_mean is not None else z
            # Cache for subsequent observations and update mean for future periods
            self._z_adj_cache[tick_key] = z_adj
            z_mean.update(z)

        # Dynamic hedge ratio
        hb = self.params.beta * (price_a / price_b)

        # Entry threshold - support both std mode and absolute dollar mode
        if self.params.use_std_mode:
            # Std mode: enter when |z_adj| >= entry_std * std
            entry_threshold = self.params.entry_std * self.params.std
            should_enter_short = z_adj >= entry_threshold
            should_enter_long = z_adj <= -entry_threshold
            reason_suffix = f'({z_adj/self.params.std:.1f}std)'
        else:
            # Dollar mode: enter when |z_adj * price_a| >= entry_abs
            dollar_mag = abs(z_adj) * price_a
            should_enter_short = z_adj > 0 and dollar_mag >= self.params.entry_abs
            should_enter_long = z_adj < 0 and dollar_mag >= self.params.entry_abs
            reason_suffix = f'(${dollar_mag:.2f})'

        # State machine
        if self.state == PositionState.FLAT:
            if should_enter_short:
                # z > 0 means a is overvalued vs b -> short a, long b
                self.state = PositionState.SHORT
                self._last_z_adj = z_adj
                self._entry_hb = hb  # Lock in raw hedge ratio
                orders = [
                    Order(a, 1, 'SELL', sec_a.get('bid', price_a)),
                    Order(b, hb, 'BUY', sec_b.get('ask', price_b)),  # Raw hb, scaled later
                ]
                return self.enter_short(orders, f'z={z_adj:.4f} {reason_suffix} hb={hb:.2f}')

            elif should_enter_long:
                # z < 0 means a is undervalued vs b -> long a, short b
                self.state = PositionState.LONG
                self._last_z_adj = z_adj
                self._entry_hb = hb  # Lock in raw hedge ratio
                orders = [
                    Order(a, 1, 'BUY', sec_a.get('ask', price_a)),
                    Order(b, hb, 'SELL', sec_b.get('bid', price_b)),  # Raw hb, scaled later
                ]
                return self.enter_long(orders, f'z={z_adj:.4f} {reason_suffix} hb={hb:.2f}')

        elif self.state == PositionState.LONG:
            # Exit when z crosses 0 (from negative to >= 0)
            if z_adj >= 0:
                self.state = PositionState.FLAT
                exit_hb = self._entry_hb  # Use same hedge ratio as entry
                self._entry_hb = 0.0
                orders = [
                    Order(a, 1, 'SELL', sec_a.get('bid', price_a)),
                    Order(b, exit_hb, 'BUY', sec_b.get('ask', price_b)),
                ]
                return self.exit(orders, f'z crossed 0: {z_adj:.4f}')

        elif self.state == PositionState.SHORT:
            # Exit when z crosses 0 (from positive to <= 0)
            if z_adj <= 0:
                self.state = PositionState.FLAT
                exit_hb = self._entry_hb  # Use same hedge ratio as entry
                self._entry_hb = 0.0
                orders = [
                    Order(a, 1, 'BUY', sec_a.get('ask', price_a)),
                    Order(b, exit_hb, 'SELL', sec_b.get('bid', price_b)),
                ]
                return self.exit(orders, f'z crossed 0: {z_adj:.4f}')

        self._last_z_adj = z_adj
        if self.params.use_std_mode:
            return self.hold(f'z={z_adj:.4f} ({z_adj/self.params.std:.1f}std)')
        else:
            dollar_mag = abs(z_adj) * price_a
            return self.hold(f'z={z_adj:.4f} (${dollar_mag:.2f})')
