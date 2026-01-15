"""Portfolio allocator for multi-signal coordination.

Weighted top-N allocation: ranks signals by strength, allocates to the
top signals that exceed the minimum threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strategies.base import Order


@dataclass
class StrategySpec:
    """Per-strategy signal input."""
    name: str
    signal: float               # Spread value (+ = short spread, - = long spread)
    sigma: float                # Normalization factor (e.g., std dev)
    build_pos: Callable[[dict], dict]  # prices → shares dict for +1 unit of short spread

    @property
    def abs_signal(self) -> float:
        return abs(self.signal)

    @property
    def direction(self) -> int:
        """Direction for position: +1 if signal > 0 (short spread), -1 if signal < 0 (long spread)."""
        return int(np.sign(self.signal)) if self.signal != 0 else 0

    @property
    def strength(self) -> float:
        """Signal strength for ranking."""
        return self.abs_signal / (self.sigma + 1e-9)


@dataclass
class AllocatorConfig:
    """Allocator configuration."""
    gross_limit: float = 50_000_000.0
    net_limit: float = 10_000_000.0
    max_shares: dict[str, int] = field(default_factory=lambda: {
        'IND': 200_000, 'AAA': 200_000, 'BBB': 200_000,
        'CCC': 200_000, 'DDD': 200_000, 'ETF': 300_000,
    })
    turnover_k: float = 50_000.0  # Max $ turnover per tick
    min_threshold: float = 0.12   # Minimum |spread| to be considered
    top_n: int = 4                # Max number of signals to allocate to

    # Risk management
    stop_loss_mult: float = 2.0      # Exit if |spread| exceeds entry * this (e.g., 2.0 = double)
    take_profit_mult: float = 0.3    # Exit if |spread| drops to entry * this (e.g., 0.3 = 70% reversion)
    max_hold_ticks: int = 300        # Force exit after this many ticks (~2.5 min at 0.5s poll)


class PortfolioAllocator:
    """Weighted top-N portfolio allocation.

    Algorithm:
    1. Filter signals by min_threshold
    2. Rank by strength = |signal| / sigma
    3. Take top N signals
    4. Allocate gross proportionally to strength
    5. Add IND to hit net target
    6. Apply constraints (gross, net, max shares, turnover)
    """

    LEGS = ['ETF', 'AAA', 'BBB', 'CCC', 'DDD', 'IND']
    EPS = 1e-9

    def __init__(self, config: AllocatorConfig | None = None, width: dict[str, float] | None = None):
        self.config = config or AllocatorConfig()
        self.width = width or {}
        self._prev_pos: dict[str, float] = {leg: 0.0 for leg in self.LEGS}

        # Compute max delta shares from turnover budget
        self._max_dshares = {}
        for leg in self.LEGS:
            w = self.width.get(leg, 0.01)
            self._max_dshares[leg] = min(
                self.config.turnover_k / w,
                self.config.max_shares.get(leg, 300_000)
            )

        # Position tracking for risk management
        # {strategy_name: {'entry_spread': float, 'entry_tick': int, 'direction': int}}
        self._positions: dict[str, dict] = {}
        self._current_tick: int = 0

    def allocate(
        self,
        specs: list[StrategySpec],
        prices: dict[str, float],
        current_pos: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Allocate positions to top signals above threshold.

        Args:
            specs: List of all strategy specs (will be filtered/ranked)
            prices: Current mid prices per ticker
            current_pos: Current positions (for turnover cap)

        Returns:
            Tuple of (target positions dict, list of active strategy names)
        """
        self._current_tick += 1

        # Build lookup for current signals
        signal_lookup = {s.name: s for s in specs}

        # 1. Apply risk management to existing positions
        forced_exits = set()
        for name, pos_info in list(self._positions.items()):
            spec = signal_lookup.get(name)
            if spec is None:
                # Strategy no longer providing signal - exit
                forced_exits.add(name)
                continue

            entry_spread = pos_info['entry_spread']
            entry_tick = pos_info['entry_tick']
            direction = pos_info['direction']
            ticks_held = self._current_tick - entry_tick

            # Current spread (same direction as entry)
            current_spread = spec.abs_signal

            # Check stop loss: spread moved against us (got bigger)
            if current_spread > entry_spread * self.config.stop_loss_mult:
                forced_exits.add(name)
                continue

            # Check take profit: spread reverted enough
            if current_spread < entry_spread * self.config.take_profit_mult:
                forced_exits.add(name)
                continue

            # Check time limit
            if ticks_held >= self.config.max_hold_ticks:
                forced_exits.add(name)
                continue

        # Remove forced exits from position tracking
        for name in forced_exits:
            if name in self._positions:
                del self._positions[name]

        # 2. Filter by minimum threshold (excluding forced exits)
        eligible = [s for s in specs
                    if s.abs_signal >= self.config.min_threshold
                    and s.name not in forced_exits]

        if not eligible:
            # No signals above threshold - flatten positions
            self._positions.clear()
            return self._flatten_positions(current_pos), []

        # 3. Rank by strength and take top N
        eligible.sort(key=lambda s: s.strength, reverse=True)
        top_specs = eligible[:self.config.top_n]
        active_names = [s.name for s in top_specs]

        # 4. Track new entries
        for spec in top_specs:
            if spec.name not in self._positions:
                self._positions[spec.name] = {
                    'entry_spread': spec.abs_signal,
                    'entry_tick': self._current_tick,
                    'direction': spec.direction,
                }

        # 5. Compute weights proportional to strength
        strengths = [s.strength for s in top_specs]
        total_strength = sum(strengths)
        weights = [s / total_strength for s in strengths]

        # 6. Allocate gross across top signals
        effective_gross = self.config.gross_limit

        pos = {leg: 0.0 for leg in self.LEGS}

        for spec, weight in zip(top_specs, weights):
            d = spec.direction

            unit_pos = spec.build_pos(prices)
            g_unit = self._gross(unit_pos, prices)

            if g_unit > 0:
                # Allocate fraction of effective gross budget to this strategy
                units = (weight * effective_gross / g_unit) * d
                for leg, shares in unit_pos.items():
                    if leg in pos:
                        pos[leg] += shares * units

        # 7. Project into constraints
        pos = self._project_to_limits(pos, prices)

        # 8. Apply turnover cap
        prev = current_pos or self._prev_pos
        pos = self._apply_turnover_cap(prev, pos)

        self._prev_pos = pos.copy()
        return pos, active_names

    def _flatten_positions(self, current_pos: dict[str, float] | None) -> dict[str, float]:
        """Gradually flatten all positions (respecting turnover cap)."""
        prev = current_pos or self._prev_pos
        target = {leg: 0.0 for leg in self.LEGS}
        return self._apply_turnover_cap(prev, target)

    def positions_to_orders(
        self,
        target: dict[str, float],
        current: dict[str, float],
        prices: dict[str, float],
        debug: bool = False,
    ) -> list['Order']:
        """Convert position deltas to orders.

        Args:
            target: Target positions
            current: Current positions
            prices: Current prices for limit prices
            debug: Print debug info

        Returns:
            List of Order objects
        """
        import logging
        from strategies.base import Order

        orders = []
        for ticker in self.LEGS:
            tgt = target.get(ticker, 0.0)
            cur = current.get(ticker, 0.0)
            delta = tgt - cur

            # Skip small changes (dead band: 10% of current or 1000 shares minimum)
            min_delta = max(1000, abs(cur) * 0.10)

            if debug:
                logging.debug('  %s: cur=%.0f tgt=%.0f delta=%.0f min_delta=%.0f %s',
                             ticker, cur, tgt, delta, min_delta,
                             'SKIP' if abs(delta) < min_delta else 'ORDER')

            if abs(delta) < min_delta:
                continue

            qty = abs(round(delta))
            if delta > 0:
                side = 'BUY'
                price = prices.get(ticker, 0) * 1.001  # Slight buffer for fills
            else:
                side = 'SELL'
                price = prices.get(ticker, 0) * 0.999

            orders.append(Order(ticker, qty, side, price))

        return orders

    def _gross(self, pos: dict[str, float], prices: dict[str, float]) -> float:
        """Compute gross exposure."""
        return sum(abs(pos.get(leg, 0) * prices.get(leg, 0)) for leg in self.LEGS)

    def _net(self, pos: dict[str, float], prices: dict[str, float]) -> float:
        """Compute net exposure."""
        return sum(pos.get(leg, 0) * prices.get(leg, 0) for leg in self.LEGS)

    def _project_to_limits(
        self,
        pos: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Project positions into constraints (max shares, gross, net)."""
        # Clip to max shares
        for leg in self.LEGS:
            max_sh = self.config.max_shares.get(leg, 300_000)
            pos[leg] = np.clip(pos.get(leg, 0), -max_sh, max_sh)

        # Scale non-IND to leave room for IND gross
        non_ind = [leg for leg in self.LEGS if leg != 'IND']
        g_ind = abs(pos.get('IND', 0) * prices.get('IND', 0))
        rem = max(0.0, self.config.gross_limit - g_ind)
        g_non = sum(abs(pos.get(leg, 0) * prices.get(leg, 0)) for leg in non_ind)

        if g_non > rem and g_non > 0:
            scale = rem / g_non
            for leg in non_ind:
                pos[leg] *= scale

        # Final net safety
        net = self._net(pos, prices)
        if abs(net) > self.config.net_limit and abs(net) > 0:
            scale = self.config.net_limit / abs(net)
            for leg in self.LEGS:
                pos[leg] *= scale

        return pos

    def _apply_turnover_cap(
        self,
        prev: dict[str, float],
        target: dict[str, float],
    ) -> dict[str, float]:
        """Apply turnover cap to limit position changes."""
        result = {}
        for leg in self.LEGS:
            p = prev.get(leg, 0.0)
            t = target.get(leg, 0.0)
            max_d = self._max_dshares.get(leg, 100_000)
            d = np.clip(t - p, -max_d, max_d)
            result[leg] = p + d
        return result

    def reset(self):
        """Reset allocator state (e.g., at start of new period)."""
        self._prev_pos = {leg: 0.0 for leg in self.LEGS}
        self._positions.clear()
        self._current_tick = 0
