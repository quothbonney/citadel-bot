from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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


def get_mid(sec: dict) -> float:
    """Extract mid price from a security dict."""
    bid = sec.get('bid', 0)
    ask = sec.get('ask', 0)
    if bid and ask:
        return (bid + ask) / 2
    return sec.get('last', 0)


class PositionState(str, Enum):
    FLAT = 'FLAT'
    LONG = 'LONG'
    SHORT = 'SHORT'


@dataclass
class Order:
    """A single order to be placed."""
    ticker: str
    quantity: float  # Can be fractional, rounded after scaling
    side: Literal['BUY', 'SELL']
    price: float  # Limit price


@dataclass
class Signal:
    """What a strategy returns each tick."""
    strategy_id: str
    action: Literal['enter_long', 'enter_short', 'exit', 'scale', 'hold']
    orders: list[Order] = field(default_factory=list)
    reason: str = ''


class SignalStrategy(ABC):
    """Base class for signal-generating strategies."""

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id
        self.state = PositionState.FLAT

    @abstractmethod
    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        """Compute trading signal for this tick. Does not execute."""
        pass

    def hold(self, reason: str = '') -> Signal:
        """Convenience: return a hold signal."""
        return Signal(self.strategy_id, 'hold', reason=reason)

    def enter_long(self, orders: list[Order], reason: str = '') -> Signal:
        return Signal(self.strategy_id, 'enter_long', orders, reason)

    def enter_short(self, orders: list[Order], reason: str = '') -> Signal:
        return Signal(self.strategy_id, 'enter_short', orders, reason)

    def exit(self, orders: list[Order], reason: str = '') -> Signal:
        return Signal(self.strategy_id, 'exit', orders, reason)

    def scale(self, orders: list[Order], reason: str = '') -> Signal:
        return Signal(self.strategy_id, 'scale', orders, reason)


class SpreadStrategy(SignalStrategy):
    """
    Abstract base for spread-based mean-reversion strategies.

    Handles:
    - Seasonality adjustment via expanding mean per tick
    - State machine: FLAT -> LONG/SHORT -> FLAT
    - Exit when spread crosses zero

    Subclasses implement:
    - compute_spread(): raw spread calculation
    - check_entry_long/short(): entry conditions
    - make_entry_orders/make_exit_orders(): order generation
    - format_entry_reason/format_hold_reason(): logging
    """

    def __init__(self, strategy_id: str) -> None:
        super().__init__(strategy_id)
        self._means: dict[int, RunningMean] = defaultdict(RunningMean)
        self._cache: dict[tuple[int, int], float] = {}

    def _adjust_for_seasonality(self, raw: float, case: dict) -> float:
        """Apply seasonality adjustment with per-tick caching."""
        period = case.get('period', 0)
        tick = case.get('tick', 0)
        tick_key = (period, tick)

        if tick_key in self._cache:
            return self._cache[tick_key]

        mean = self._means[tick]
        prev = mean.get()
        adj = raw - prev if prev is not None else raw
        self._cache[tick_key] = adj
        mean.update(raw)
        return adj

    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        raw = self.compute_spread(portfolio, case)
        if raw is None:
            return self.hold('missing data')

        spread_adj = self._adjust_for_seasonality(raw, case)

        if self.state == PositionState.FLAT:
            if self.check_entry_short(spread_adj):
                self.state = PositionState.SHORT
                self.on_entry()
                orders = self.make_entry_orders(portfolio, is_long=False)
                return self.enter_short(orders, self.format_entry_reason(spread_adj))
            elif self.check_entry_long(spread_adj):
                self.state = PositionState.LONG
                self.on_entry()
                orders = self.make_entry_orders(portfolio, is_long=True)
                return self.enter_long(orders, self.format_entry_reason(spread_adj))

        elif self.state == PositionState.LONG:
            if spread_adj >= 0:
                self.state = PositionState.FLAT
                orders = self.make_exit_orders(portfolio, is_long=True)
                self.on_exit()
                return self.exit(orders, f'spread crossed 0: {spread_adj:.4f}')

        elif self.state == PositionState.SHORT:
            if spread_adj <= 0:
                self.state = PositionState.FLAT
                orders = self.make_exit_orders(portfolio, is_long=False)
                self.on_exit()
                return self.exit(orders, f'spread crossed 0: {spread_adj:.4f}')

        return self.hold(self.format_hold_reason(spread_adj))

    # --- Abstract methods: subclasses must implement ---

    @abstractmethod
    def compute_spread(self, portfolio: dict, case: dict) -> float | None:
        """Compute raw spread value. Return None if data is missing."""

    @abstractmethod
    def check_entry_long(self, spread_adj: float) -> bool:
        """Return True if should enter long position."""

    @abstractmethod
    def check_entry_short(self, spread_adj: float) -> bool:
        """Return True if should enter short position."""

    @abstractmethod
    def make_entry_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        """Generate orders for entering a position."""

    @abstractmethod
    def make_exit_orders(self, portfolio: dict, is_long: bool) -> list[Order]:
        """Generate orders for exiting a position."""

    @abstractmethod
    def format_entry_reason(self, spread_adj: float) -> str:
        """Format reason string for entry signal."""

    @abstractmethod
    def format_hold_reason(self, spread_adj: float) -> str:
        """Format reason string for hold signal."""

    # --- Hooks: override if needed ---

    def on_entry(self) -> None:
        """Called when entering a position. Override to save state."""
        pass

    def on_exit(self) -> None:
        """Called when exiting a position. Override to clear state."""
        pass
