import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from params import PairCointParams, StrategyParams
from strategies.base import get_mid


@dataclass
class SpreadStats:
    """Statistics for a spread over time."""
    current: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    mean: float = 0.0
    std: float = 0.0
    count: int = 0
    
    # Recent values for rolling window
    recent_values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, value: float) -> None:
        """Update statistics with a new spread value."""
        self.current = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.count += 1
        self.recent_values.append(value)
        
        # Update mean using online algorithm
        self.mean += (value - self.mean) / self.count
        
        # Update std using online algorithm (Welford's method)
        if self.count > 1:
            # For simplicity, recalculate std from recent values
            if len(self.recent_values) > 1:
                recent_mean = sum(self.recent_values) / len(self.recent_values)
                variance = sum((x - recent_mean) ** 2 for x in self.recent_values) / len(self.recent_values)
                self.std = math.sqrt(variance)
            else:
                # Fallback: use expanding window
                variance = sum((x - self.mean) ** 2 for x in self.recent_values) / len(self.recent_values)
                self.std = math.sqrt(variance) if len(self.recent_values) > 1 else 0.0


@dataclass
class SpreadMonitor:
    """Monitors spread statistics for configured pairs."""
    
    params: StrategyParams
    stats: dict[str, SpreadStats] = field(default_factory=dict)
    log_interval: int = 10  # Log every N ticks
    
    def __post_init__(self) -> None:
        """Initialize monitors for all enabled pair cointegration strategies."""
        for pair in self.params.pair_coint:
            if pair.enabled:
                self.stats[pair.strategy_id] = SpreadStats()
                logging.info('Initialized spread monitor for %s', pair.strategy_id)
    
    def _compute_spread(self, portfolio: dict, pair: PairCointParams) -> Optional[float]:
        """Compute raw spread for a pair."""
        a, b = pair.a, pair.b
        
        sec_a = portfolio.get(a, {})
        sec_b = portfolio.get(b, {})
        if not sec_a or not sec_b:
            return None
        
        price_a = get_mid(sec_a)
        price_b = get_mid(sec_b)
        if price_a <= 0 or price_b <= 0:
            return None
        
        # Compute spread: z = log(a) - (c + beta * log(b))
        return math.log(price_a) - (pair.c + pair.beta * math.log(price_b))
    
    def update(self, portfolio: dict, case: dict) -> None:
        """Update spread statistics for all monitored pairs."""
        period = case.get('period', 0)
        tick = case.get('tick', 0)
        
        for pair in self.params.pair_coint:
            if pair.enabled and pair.strategy_id in self.stats:
                spread = self._compute_spread(portfolio, pair)
                if spread is not None:
                    self.stats[pair.strategy_id].update(spread)
                else:
                    logging.debug('No spread data available for %s at tick %d', pair.strategy_id, tick)
    
    def log_stats(self, case: dict) -> None:
        """Log current spread statistics."""
        tick = case.get('tick', 0)
        
        # Only log at specified intervals
        if tick % self.log_interval != 0:
            return
        
        for pair_id, stats in self.stats.items():
            if stats.count > 0:
                # Calculate z-score for context
                z_score = (stats.current - stats.mean) / stats.std if stats.std > 0 else 0.0
                logging.info(
                    '[RISK MONITOR] %s: spread=%.6f (z=%.2f) | mean=%.6f | std=%.6f | range=[%.6f, %.6f] | count=%d',
                    pair_id, stats.current, z_score, stats.mean, stats.std, stats.min, stats.max, stats.count
                )
    
    def get_stats(self, pair_id: str) -> Optional[SpreadStats]:
        """Get statistics for a specific pair."""
        return self.stats.get(pair_id)
    
    def get_summary(self) -> dict:
        """Get summary of all monitored spreads."""
        summary = {}
        for pair_id, stats in self.stats.items():
            if stats.count > 0:
                summary[pair_id] = {
                    'current': stats.current,
                    'mean': stats.mean,
                    'std': stats.std,
                    'min': stats.min,
                    'max': stats.max,
                    'count': stats.count,
                }
        return summary

