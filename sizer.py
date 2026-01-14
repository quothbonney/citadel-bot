from typing import Protocol

from market import Market


class Sizer(Protocol):
    """Protocol for position sizing."""

    def scale(self, base_qty: int, portfolio: dict, market: Market) -> int:
        """Scale base quantity based on risk/portfolio state."""
        ...


class UnitSizer:
    """Default sizer: returns base quantity unchanged (1x)."""

    def scale(self, base_qty: int, portfolio: dict, market: Market) -> int:
        return base_qty


class FixedSizer:
    """Fixed multiplier sizer."""

    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier

    def scale(self, base_qty: int, portfolio: dict, market: Market) -> int:
        return base_qty * self.multiplier
