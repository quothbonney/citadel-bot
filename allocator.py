"""Portfolio allocator for multi-signal coordination.

Aggregates signals from multiple strategies and allocates gross budget
proportionally to |signal|/(sigma+eps), using IND as net exposure knob.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strategies.base import Order


@dataclass
class StrategySpec:
    """Per-strategy allocation input."""
    name: str
    signal: float               # Scalar signal (+ = short spread, - = long spread)
    sigma: float                # Risk proxy (entry threshold)
    build_pos: Callable[[dict], dict]  # prices → shares dict for +1 unit
    direction: int = 0          # +1 = long spread, -1 = short spread

    def __post_init__(self):
        self.direction = int(np.sign(self.signal)) if self.signal != 0 else 0


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
    net_mode: str = 'signal'  # 'signal' or 'momentum'


class PortfolioAllocator:
    """Multi-signal portfolio allocation with constraints.

    Algorithm:
    1. Compute weights: w_i = |signal_i| / (sigma_i + eps)
    2. Allocate gross: units_i = (w_i / sum(w)) * gross_limit / g_unit_i
    3. Build combined position (non-IND legs)
    4. Add IND to hit net target
    5. Project into constraints (gross, net, max shares)
    6. Apply turnover cap
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
    ) -> dict[str, float]:
        """Allocate positions across active strategies.

        Args:
            specs: List of active strategy specs
            prices: Current mid prices per ticker
            current_pos: Current positions (for turnover cap)

        Returns:
            Target positions dict[ticker, shares]
        """
        if not specs:
            return self._prev_pos.copy()

        # 1. Compute weights: w_i = |signal_i| / (sigma_i + eps)
        weights = []
        for spec in specs:
            w = abs(spec.signal) / (spec.sigma + self.EPS)
            weights.append(w)

        w_sum = sum(weights)
        if w_sum < self.EPS:
            return self._prev_pos.copy()

        w_frac = [w / w_sum for w in weights]

        # 2. Allocate gross across strategies
        pos = {leg: 0.0 for leg in self.LEGS}
        agg_dir = 0.0

        for spec, frac in zip(specs, w_frac):
            d = spec.direction
            agg_dir += frac * d

            unit_pos = spec.build_pos(prices)
            g_unit = self._gross(unit_pos, prices)

            if g_unit > 0:
                # Allocate fraction of gross budget to this strategy
                units = (frac * self.config.gross_limit / g_unit) * d
                for leg, shares in unit_pos.items():
                    if leg in pos:
                        pos[leg] += shares * units

        # 3. Add IND to hit net target
        net_target = np.sign(agg_dir) * self.config.net_limit if agg_dir != 0 else 0.0
        net_now = self._net(pos, prices)
        ind_price = prices.get('IND', 1.0)
        if ind_price > 0:
            ind_delta = (net_target - net_now) / ind_price
            pos['IND'] += ind_delta

        # 4. Project into constraints
        pos = self._project_to_limits(pos, prices)

        # 5. Apply turnover cap
        prev = current_pos or self._prev_pos
        pos = self._apply_turnover_cap(prev, pos)

        self._prev_pos = pos.copy()
        return pos

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

            if abs(delta) < 0.5:  # Skip tiny changes
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
