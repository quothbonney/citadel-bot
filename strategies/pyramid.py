from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from params import PyramidParams


class PyramidMixin:
    """Shared pyramid state and logic for any spread strategy.

    Provides level-based entry/scaling/exit logic that can be mixed into
    any SpreadStrategy subclass.

    State attributes (initialized by _init_pyramid):
        _triggered_levels: Which entry levels have been triggered
        _current_size: Total units currently held
        _hold_ticks: Ticks held in current position
    """

    _triggered_levels: list[bool]
    _current_size: int
    _hold_ticks: int

    def _init_pyramid(self, pyramid: PyramidParams) -> None:
        """Initialize pyramid state. Call in __init__."""
        self._triggered_levels = [False] * pyramid.max_level
        self._current_size = 0
        self._hold_ticks = 0

    def _reset_pyramid(self, pyramid: PyramidParams) -> None:
        """Reset pyramid state on position exit."""
        self._triggered_levels = [False] * pyramid.max_level
        self._current_size = 0
        self._hold_ticks = 0

    def _enter_at_level(self, level_idx: int, pyramid: PyramidParams) -> int:
        """Mark level as triggered and return size to add.

        Returns:
            Size to add for this level
        """
        size = pyramid.entry_sizes[level_idx]
        self._triggered_levels[level_idx] = True
        self._current_size = size
        self._hold_ticks = 0
        return size

    def _check_scale_up(self, abs_spread: float, pyramid: PyramidParams) -> tuple[int, int] | None:
        """Check if we should scale up at a new entry level.

        Args:
            abs_spread: Absolute value of adjusted spread
            pyramid: Pyramid parameters

        Returns:
            (level_idx, size) if scaling, None otherwise
        """
        for i, (level, size) in enumerate(zip(pyramid.entry_levels, pyramid.entry_sizes)):
            if not self._triggered_levels[i] and abs_spread >= level:
                self._triggered_levels[i] = True
                self._current_size += size
                return (i, size)
        return None

    def _check_exit(self, spread_adj: float, pyramid: PyramidParams, is_long: bool) -> float | None:
        """Check if spread has crossed an exit level.

        Args:
            spread_adj: Adjusted spread value
            pyramid: Pyramid parameters
            is_long: True if in long position

        Returns:
            Exit level that was crossed, or None
        """
        for exit_level in pyramid.exit_levels:
            if is_long and spread_adj >= exit_level:
                return exit_level
            elif not is_long and spread_adj <= exit_level:
                return exit_level
        return None

    def _check_stop_loss(self, abs_spread: float, pyramid: PyramidParams) -> bool:
        """Check if stop loss was hit.

        Returns:
            True if stop loss triggered
        """
        return pyramid.stop_loss is not None and abs_spread >= pyramid.stop_loss
