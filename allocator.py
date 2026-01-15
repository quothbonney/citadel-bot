"""Portfolio allocator for multi-signal coordination.

This allocator builds a combined target portfolio from multiple strategy signals,
then projects that portfolio into global constraints (gross/net/max shares) and
applies a turnover cap.

Allocation objective: maximize expected *net edge* (mean-reversion alpha proxy
minus estimated costs) with an explicit L1 switching penalty on weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategies.base import Order


@dataclass
class Signal:
    """A strategy's signal for allocation."""
    name: str
    s_dollars: float          # Dollarized spread/mispricing (+ = short spread, - = long spread)
    entry_dollars: float      # Entry threshold in dollars (>= 0)
    rt_cost_dollars: float    # Estimated round-trip cost (per 1 unit) in dollars (>= 0)
    legs: dict[str, float]  # Position per unit: {ticker: shares} for +1 unit SHORT spread

    @property
    def direction(self) -> int:
        """Trade direction: +1 if s_dollars > 0 (short spread), -1 if s_dollars < 0 (long spread)."""
        if self.s_dollars > 0:
            return 1
        elif self.s_dollars < 0:
            return -1
        return 0


@dataclass
class AllocatorConfig:
    """Allocator configuration."""
    gross_limit: float = 50_000_000.0
    net_limit: float = 10_000_000.0
    max_shares: dict[str, int] = field(default_factory=lambda: {
        'IND': 200_000, 'AAA': 200_000, 'BBB': 200_000,
        'CCC': 200_000, 'DDD': 200_000, 'ETF': 300_000,
    })
    top_n: int = 4                   # Max signals to allocate to (active universe size)
    turnover_pct: float = 0.10       # Max position change per tick as fraction of max shares
    min_threshold: float = 0.0       # Minimum |S| (in dollars) to even consider (extra filter)

    # Edge model
    horizon_bars: int = 10
    switch_lambda: float = 0.10      # L1 penalty in weight space
    regime_cutoff: float = 2.5       # Kill edge when sigma_hat / median(sigma_hat) > cutoff
    w_max: float = 1.0               # Upper bound per-weight (<= 1)

    # Portfolio vol scaling
    vol_scale_enabled: bool = False  # Enable portfolio vol scaling
    target_vol: float = 50_000.0     # Target portfolio vol ($ per tick std dev)
    vol_halflife: int = 20           # EWMA halflife for realized vol
    
    # Asymmetric turnover: allow faster exits than entries
    exit_turnover_mult: float = 5.0  # Exit turnover = turnover_pct * this multiplier
    
    # Drawdown throttle
    dd_throttle_enabled: bool = False
    dd_throttle_threshold: float = 100_000.0  # $ drawdown to trigger throttle
    dd_throttle_factor: float = 0.5           # Reduce turnover by this factor when in DD


TICKERS = ('AAA', 'BBB', 'CCC', 'DDD', 'ETF', 'IND')


class Allocator:
    """Weighted portfolio allocation across multiple signals.
    
    Algorithm:
    - Compute per-signal net edge:
        edge = max(0, |S| - entry - rt_cost) / sigma_hat
      where sigma_hat is EWMA volatility of ΔS (spread changes).
    - Apply regime filter: if sigma_hat/median(sigma_hat) > regime_cutoff => edge = 0.
    - Choose weights w by maximizing:
        sum_j w_j * edge_j - lambda * sum_j |w_j - wprev_j|
      subject to w in simplex, and 0 <= w_j <= w_max.
    - Allocate gross across signals by weights and per-signal unit_gross.
    - Project to constraints and apply turnover cap.
    """

    def __init__(self, config: AllocatorConfig | None = None) -> None:
        self.config = config or AllocatorConfig()
        self._prev_pos: dict[str, float] = {t: 0.0 for t in TICKERS}
        self._prev_w: dict[str, float] = {}
        self._last_s: dict[str, float] = {}
        self._var_s: dict[str, float] = {}  # EWMA variance of ΔS per signal
        self._last_diag: dict[str, object] = {}
        self._last_inputs: dict[str, dict[str, float]] = {}
        
        # Portfolio vol scaling state
        self._last_portfolio_value: float | None = None
        self._ewma_var: float = 0.0  # EWMA variance of portfolio returns
        self._vol_scale: float = 1.0  # Current vol scaling factor
        self._vol_tick_count: int = 0  # Ticks since vol tracking started
        
        # Drawdown throttle state
        self._peak_pnl: float = 0.0
        self._current_pnl: float = 0.0
        self._dd_throttle_active: bool = False

    def allocate(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        current_pos: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Compute target positions from signals.
        
        Returns:
            (target_positions, active_strategy_names)
        """
        # Update portfolio vol scaling
        if current_pos is not None:
            self._update_vol_scale(current_pos, prices)
        
        if not signals:
            self._last_diag = {"reason": "no_signals", "vol_scale": self._vol_scale}
            return self._flatten(current_pos), []

        # Optional coarse filter
        signals = [s for s in signals if abs(s.s_dollars) >= self.config.min_threshold]
        if not signals:
            self._last_diag = {"reason": "below_min_threshold", "vol_scale": self._vol_scale}
            return self._flatten(current_pos), []

        self._last_inputs = {
            s.name: {"s_dollars": float(s.s_dollars), "entry_dollars": float(s.entry_dollars), "rt_cost_dollars": float(s.rt_cost_dollars)}
            for s in signals
        }

        # Compute edges for all provided signals
        edges, sigma_hat, median_sigma, regime_ratio, net_raw = self._compute_edges(signals)
        if not edges:
            self._last_diag = {
                "reason": "no_positive_edges",
                "edges": {},
                "sigma_hat": sigma_hat,
                "median_sigma": median_sigma,
                "regime_ratio": regime_ratio,
                "net_raw": net_raw,
                "inputs": self._last_inputs,
            }
            return self._flatten(current_pos), []

        # Candidate set: include top-N by edge plus top-N by previous weight (stability).
        by_edge = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)
        top_edge = [name for name, _ in by_edge[: self.config.top_n]]

        by_prev = sorted(self._prev_w.items(), key=lambda kv: kv[1], reverse=True)
        top_prev = [name for name, _ in by_prev[: self.config.top_n]]

        candidates = {n for n in top_edge + top_prev if n in edges}
        if not candidates:
            return self._flatten(current_pos), []

        # Optimize weights on candidate set only
        cand_edges = {n: edges[n] for n in candidates}
        wprev = {n: self._prev_w.get(n, 0.0) for n in candidates}
        w = self._optimize_weights(edges=cand_edges, wprev=wprev)

        active_names = [n for n, ww in sorted(w.items(), key=lambda kv: kv[1], reverse=True) if ww > 0]

        # Allocate: each signal gets weight fraction of gross budget
        # Apply vol scaling to gross limit
        effective_gross = self.config.gross_limit * self._vol_scale
        
        pos = {t: 0.0 for t in TICKERS}

        sig_by_name = {s.name: s for s in signals}
        for name, weight in w.items():
            if weight <= 0:
                continue
            sig = sig_by_name.get(name)
            if sig is None:
                continue
            # Compute $ value per unit for this signal's legs
            unit_gross = sum(abs(sig.legs.get(t, 0) * prices.get(t, 0)) for t in TICKERS)
            if unit_gross <= 0:
                continue
            
            # How many units to allocate (in signal direction)
            budget = weight * effective_gross
            units = (budget / unit_gross) * sig.direction
            
            # Add to aggregate position
            for t, leg_shares in sig.legs.items():
                if t in pos:
                    pos[t] += leg_shares * units
        
        # Project into constraints
        pos = self._project(pos, prices)
        
        # Apply turnover cap - use current_pos as ground truth, NOT _prev_pos
        prev = current_pos or {t: 0.0 for t in TICKERS}
        pos = self._cap_turnover(prev, pos)
        
        self._prev_w = w
        self._last_diag = {
            "reason": "ok",
            "active": active_names,
            "weights": w,
            "edges": edges,
            "sigma_hat": sigma_hat,
            "median_sigma": median_sigma,
            "regime_ratio": regime_ratio,
            "net_raw": net_raw,
            "inputs": self._last_inputs,
            "vol_scale": self._vol_scale,
            "effective_gross": effective_gross,
        }
        return pos, active_names
    
    def _update_vol_scale(self, current_pos: dict[str, float], prices: dict[str, float]) -> None:
        """Update portfolio vol scaling factor based on EWMA of portfolio value changes."""
        if not self.config.vol_scale_enabled:
            self._vol_scale = 1.0
            return
        
        # Compute current portfolio value (net exposure)
        portfolio_value = sum(current_pos.get(t, 0) * prices.get(t, 0) for t in TICKERS)
        
        # Initialize on first tick with non-zero positions
        if self._last_portfolio_value is None:
            if abs(portfolio_value) > 1e-6:
                self._last_portfolio_value = portfolio_value
                self._vol_tick_count = 1
            return
        
        # Compute return (change in portfolio value)
        ret = portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = portfolio_value
        self._vol_tick_count += 1
        
        # Warmup period: need at least 2x halflife ticks before applying vol scaling
        warmup_ticks = self.config.vol_halflife * 2
        if self._vol_tick_count < warmup_ticks:
            # Still warming up - update EWMA but keep scale at 1.0
            halflife = max(1, self.config.vol_halflife)
            alpha = 1.0 - np.exp(-np.log(2) / halflife)
            self._ewma_var = (1.0 - alpha) * self._ewma_var + alpha * (ret * ret)
            self._vol_scale = 1.0
            return
        
        # Update EWMA variance using halflife
        # alpha = 1 - exp(-ln(2) / halflife)
        halflife = max(1, self.config.vol_halflife)
        alpha = 1.0 - np.exp(-np.log(2) / halflife)
        
        self._ewma_var = (1.0 - alpha) * self._ewma_var + alpha * (ret * ret)
        realized_vol = float(np.sqrt(self._ewma_var))
        
        # Compute vol scale: target_vol / realized_vol, clamped to [0.25, 1.0]
        if realized_vol > 0:
            raw_scale = self.config.target_vol / realized_vol
            self._vol_scale = float(np.clip(raw_scale, 0.25, 1.0))
        else:
            self._vol_scale = 1.0
    
    def sync_positions(self, actual_pos: dict[str, float]) -> None:
        """Sync internal state with actual positions after execution."""
        for t in TICKERS:
            self._prev_pos[t] = actual_pos.get(t, 0.0)
    
    def update_pnl(self, current_pnl: float) -> None:
        """Update PnL tracking for drawdown throttle."""
        self._current_pnl = current_pnl
        if current_pnl > self._peak_pnl:
            self._peak_pnl = current_pnl
        
        # Check if drawdown throttle should be active
        if self.config.dd_throttle_enabled:
            drawdown = self._peak_pnl - self._current_pnl
            self._dd_throttle_active = drawdown > self.config.dd_throttle_threshold

    def _flatten(self, current_pos: dict[str, float] | None) -> dict[str, float]:
        """Move toward zero, respecting turnover cap."""
        prev = current_pos or self._prev_pos
        target = {t: 0.0 for t in TICKERS}
        return self._cap_turnover(prev, target)

    def _project(self, pos: dict[str, float], prices: dict[str, float]) -> dict[str, float]:
        """Project positions into constraint space."""
        # Clip to max shares
        for t in TICKERS:
            max_sh = self.config.max_shares.get(t, 200_000)
            pos[t] = np.clip(pos.get(t, 0), -max_sh, max_sh)
        
        # Scale down if over gross limit
        gross = sum(abs(pos.get(t, 0) * prices.get(t, 0)) for t in TICKERS)
        if gross > self.config.gross_limit:
            scale = self.config.gross_limit / gross
            for t in TICKERS:
                pos[t] *= scale
        
        # Scale down if over net limit
        net = sum(pos.get(t, 0) * prices.get(t, 0) for t in TICKERS)
        if abs(net) > self.config.net_limit:
            scale = self.config.net_limit / abs(net)
            for t in TICKERS:
                pos[t] *= scale
        
        return pos

    def _cap_turnover(
        self,
        prev: dict[str, float],
        target: dict[str, float],
    ) -> dict[str, float]:
        """Limit position changes per tick.
        
        Supports:
        - Asymmetric turnover: exits (moves toward 0) can be faster than entries
        - Drawdown throttle: reduce entry turnover when in drawdown
        """
        base_pct = self.config.turnover_pct
        exit_mult = self.config.exit_turnover_mult
        
        # Apply drawdown throttle to entry turnover (not exits - we still want fast exits)
        entry_pct = base_pct
        if self._dd_throttle_active:
            entry_pct *= self.config.dd_throttle_factor
        
        result = {}
        for t in TICKERS:
            p = prev.get(t, 0.0)
            tgt = target.get(t, 0.0)
            max_sh = self.config.max_shares.get(t, 200_000)
            
            # Determine if this is an exit (moving toward 0) or entry (moving away from 0)
            is_exit = abs(tgt) < abs(p)
            
            # Use higher turnover for exits, throttled turnover for entries in DD
            pct = base_pct * exit_mult if is_exit else entry_pct
            max_delta = max_sh * pct
            
            delta = np.clip(tgt - p, -max_delta, max_delta)
            result[t] = p + delta
        
        return result

    def _compute_edges(self, signals: list[Signal]) -> tuple[dict[str, float], dict[str, float], float, dict[str, float], dict[str, float]]:
        """Compute net edge scores and sigma_hat for each signal.

        Returns:
            edges: positive edge scores
            sigma_hat: EWMA sigma of ΔS
            median_sigma: median of sigma_hat over positive sigmas
            regime_ratio: sigma_hat / median_sigma (0 if median_sigma==0)
            net_raw: max(0, |S| - entry - rt_cost) (in dollars)
        """
        h = self.config.horizon_bars
        if h <= 0:
            raise ValueError("horizon_bars must be > 0")
        alpha = 2.0 / (h + 1.0)

        sigma_hat: dict[str, float] = {}
        # Update EWMA sigma_hat per signal
        for s in signals:
            last = self._last_s.get(s.name)
            if last is None:
                self._last_s[s.name] = s.s_dollars
                self._var_s[s.name] = 0.0
                sigma_hat[s.name] = 0.0
                continue

            d = s.s_dollars - last
            var = self._var_s.get(s.name, 0.0)
            var = (1.0 - alpha) * var + alpha * (d * d)
            self._var_s[s.name] = var
            self._last_s[s.name] = s.s_dollars
            sigma_hat[s.name] = float(np.sqrt(var))

        # Median sigma for regime filter (only consider > 0)
        positive_sigmas = sorted(v for v in sigma_hat.values() if v > 0)
        median_sigma = 0.0
        if positive_sigmas:
            mid = len(positive_sigmas) // 2
            if len(positive_sigmas) % 2:
                median_sigma = positive_sigmas[mid]
            else:
                median_sigma = 0.5 * (positive_sigmas[mid - 1] + positive_sigmas[mid])

        edges: dict[str, float] = {}
        regime_ratio: dict[str, float] = {}
        net_raw: dict[str, float] = {}
        for s in signals:
            sig = sigma_hat.get(s.name, 0.0)
            if sig <= 0:
                regime_ratio[s.name] = 0.0
                net_raw[s.name] = 0.0
                continue

            ratio = (sig / median_sigma) if median_sigma > 0 else 0.0
            regime_ratio[s.name] = ratio

            net = abs(s.s_dollars) - s.entry_dollars - s.rt_cost_dollars
            net_raw[s.name] = max(0.0, net)

            if median_sigma > 0 and ratio > self.config.regime_cutoff:
                continue

            if net <= 0:
                continue

            edges[s.name] = net / sig

        return edges, sigma_hat, median_sigma, regime_ratio, net_raw

    def diagnostics(self) -> dict[str, object]:
        """Return last allocator diagnostics (for dashboards/debugging)."""
        return dict(self._last_diag)

    def _optimize_weights(self, edges: dict[str, float], wprev: dict[str, float]) -> dict[str, float]:
        """Solve L1-penalized weight update over a small candidate set.

        maximize  sum_j w_j * edge_j - lambda * sum_j |w_j - wprev_j|
        subject to sum_j w_j = 1, 0 <= w_j <= w_max
        """
        names = list(edges.keys())
        if not names:
            return {}

        # If all edges are non-positive, don't allocate.
        if max(edges.values()) <= 0:
            return {n: 0.0 for n in names}

        w_max = self.config.w_max
        if w_max <= 0 or w_max > 1.0:
            raise ValueError("w_max must be in (0, 1]")

        # Normalize/clip previous weights onto current name set
        wprev_vec = {n: max(0.0, float(wprev.get(n, 0.0))) for n in names}
        s_prev = sum(wprev_vec.values())
        if s_prev > 0:
            for n in names:
                wprev_vec[n] /= s_prev
        else:
            # No prior -> start from uniform (only affects penalty constants)
            for n in names:
                wprev_vec[n] = 1.0 / len(names)

        lam = self.config.switch_lambda
        if lam < 0:
            raise ValueError("switch_lambda must be >= 0")

        best_obj = -float("inf")
        best_w: dict[str, float] | None = None

        # Enumerate sign patterns for (w - wprev): s=+1 => w>=wprev, s=-1 => w<=wprev.
        # Small universe (<= 2*top_n) => cheap exact solve by pattern enumeration.
        for mask in range(1 << len(names)):
            lb: dict[str, float] = {}
            ub: dict[str, float] = {}
            coeff: dict[str, float] = {}

            for i, n in enumerate(names):
                wp = wprev_vec[n]
                sgn = 1.0 if (mask & (1 << i)) else -1.0

                if sgn > 0:
                    lb[n] = wp
                    ub[n] = w_max
                else:
                    lb[n] = 0.0
                    ub[n] = min(w_max, wp)

                coeff[n] = edges[n] - lam * sgn

            sum_lb = sum(lb.values())
            sum_ub = sum(ub.values())
            if sum_lb - 1.0 > 1e-12:
                continue
            if 1.0 - sum_ub > 1e-12:
                continue

            w = dict(lb)
            remaining = 1.0 - sum_lb

            # Greedy fill by coefficient
            for n in sorted(names, key=lambda x: coeff[x], reverse=True):
                if remaining <= 1e-15:
                    break
                cap = ub[n] - w[n]
                if cap <= 0:
                    continue
                add = cap if cap < remaining else remaining
                w[n] += add
                remaining -= add

            if abs(remaining) > 1e-9:
                continue

            # Compute full objective (include constants for correctness across patterns)
            obj = 0.0
            for n in names:
                obj += w[n] * edges[n] - lam * abs(w[n] - wprev_vec[n])

            if obj > best_obj:
                best_obj = obj
                best_w = w

        if best_w is None:
            # Should not happen for sane inputs; fall back to best-edge one-hot.
            best = max(names, key=lambda n: edges[n])
            return {n: (1.0 if n == best else 0.0) for n in names}

        return best_w

    def to_orders(
        self,
        target: dict[str, float],
        current: dict[str, float],
        prices: dict[str, float],
        min_delta: int = 1,
    ) -> list[Order]:
        """Convert position delta to orders."""
        orders = []
        
        for t in TICKERS:
            tgt = target.get(t, 0.0)
            cur = current.get(t, 0.0)
            delta = tgt - cur
            
            if abs(delta) < min_delta:
                continue
            
            qty = abs(round(delta))
            price = prices.get(t, 0)
            
            if delta > 0:
                orders.append(Order(t, qty, 'BUY', price * 1.001))
            else:
                orders.append(Order(t, qty, 'SELL', price * 0.999))
        
        return orders

    def reset(self) -> None:
        """Reset state (call at period boundaries)."""
        self._prev_pos = {t: 0.0 for t in TICKERS}


# Legacy compatibility aliases
StrategySpec = Signal
PortfolioAllocator = Allocator
