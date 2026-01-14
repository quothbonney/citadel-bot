from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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
    action: Literal['enter_long', 'enter_short', 'exit', 'hold']
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
