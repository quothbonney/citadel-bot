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
        # 1. Filter by minimum threshold
        eligible = [s for s in specs if s.abs_signal >= self.config.min_threshold]

        if not eligible:
            # No signals above threshold - flatten positions
            return self._flatten_positions(current_pos), []

        # 2. Rank by strength and take top N
        eligible.sort(key=lambda s: s.strength, reverse=True)
        top_specs = eligible[:self.config.top_n]
        active_names = [s.name for s in top_specs]

        # 3. Compute weights proportional to strength
        strengths = [s.strength for s in top_specs]
        total_strength = sum(strengths)
        weights = [s / total_strength for s in strengths]

        # 4. Allocate gross across top signals
        pos = {leg: 0.0 for leg in self.LEGS}
        agg_dir = 0.0

        for spec, weight in zip(top_specs, weights):
            d = spec.direction
            agg_dir += weight * d

            unit_pos = spec.build_pos(prices)
            g_unit = self._gross(unit_pos, prices)

            if g_unit > 0:
                # Allocate fraction of gross budget to this strategy
                units = (weight * self.config.gross_limit / g_unit) * d
                for leg, shares in unit_pos.items():
                    if leg in pos:
                        pos[leg] += shares * units

        # 5. Add IND to hit net target
        net_target = np.sign(agg_dir) * self.config.net_limit if agg_dir != 0 else 0.0
        net_now = self._net(pos, prices)
        ind_price = prices.get('IND', 1.0)
        if ind_price > 0:
            ind_delta = (net_target - net_now) / ind_price
            pos['IND'] += ind_delta

        # 6. Project into constraints
        pos = self._project_to_limits(pos, prices)

        # 7. Apply turnover cap
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
    ) -> list['Order']:
        """Convert position deltas to orders.

        Args:
            target: Target positions
            current: Current positions
            prices: Current prices for limit prices

        Returns:
            List of Order objects
        """
        from strategies.base import Order

        orders = []
        for ticker in self.LEGS:
            tgt = target.get(ticker, 0.0)
            cur = current.get(ticker, 0.0)
            delta = tgt - cur

            # Skip small changes (dead band: 15% of current or 5000 shares minimum)
            min_delta = max(5000, abs(cur) * 0.15)
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
