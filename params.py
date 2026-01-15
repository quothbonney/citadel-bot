import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PyramidParams:
    """Level-based entry/exit configuration for spread strategies."""
    entry_levels: tuple[float, ...]   # Spread thresholds for entering/scaling
    entry_sizes: tuple[int, ...]      # Size to add at each level
    exit_levels: tuple[float, ...]    # Spread thresholds for exiting
    stop_loss: float | None = None    # Hard stop if |spread| exceeds this

    @property
    def max_level(self) -> int:
        """Number of entry levels."""
        return len(self.entry_levels)

    @property
    def total_size(self) -> int:
        """Total position size when fully scaled."""
        return sum(self.entry_sizes)

    @property
    def first_entry(self) -> float:
        """First entry threshold."""
        return self.entry_levels[0]


def _parse_pyramid(data: dict) -> PyramidParams:
    """Parse pyramid config from dict."""
    return PyramidParams(
        entry_levels=tuple(data['entry_levels']),
        entry_sizes=tuple(data['entry_sizes']),
        exit_levels=tuple(data['exit_levels']),
        stop_loss=data.get('stop_loss'),
    )


@dataclass(frozen=True)
class PairCointParams:
    """Parameters for a pair cointegration strategy."""
    a: str  # Long leg ticker
    b: str  # Hedge leg ticker
    c: float  # Intercept from regression
    beta: float  # Cointegration coefficient
    std: float  # Standard deviation of z
    pyramid: PyramidParams  # Entry/exit levels and sizes
    enabled: bool = True

    @property
    def strategy_id(self) -> str:
        return f'pair_{self.a}_{self.b}'


@dataclass(frozen=True)
class EtfNavParams:
    """Parameters for ETF-NAV arbitrage strategy."""
    pyramid: PyramidParams  # Entry/exit levels and sizes
    eod_flat: bool = False  # Flatten at end of day (tick 390)
    enabled: bool = True

    @property
    def strategy_id(self) -> str:
        return 'etf_nav'


@dataclass(frozen=True)
class AllocatorConfig:
    """Portfolio allocator configuration."""
    gross_limit: float = 50_000_000.0
    net_limit: float = 10_000_000.0
    max_shares: dict[str, int] | None = None
    turnover_pct: float = 0.10       # Max position change per tick as fraction of max_shares
    min_threshold: float = 0.0       # Minimum |S$| to even consider (extra filter, 0 = disabled)
    top_n: int = 4                   # Max signals to allocate to
    horizon_bars: int = 10           # EWMA horizon for sigma_hat
    switch_lambda: float = 0.10      # L1 switching penalty in weight space
    regime_cutoff: float = 2.5       # Kill edge when sigma_hat/median > cutoff
    w_max: float = 1.0               # Max weight per signal (0 < w_max <= 1)
    vol_scale_enabled: bool = False  # Enable portfolio vol scaling
    target_vol: float = 50_000.0     # Target portfolio vol ($ per tick std dev)
    vol_halflife: int = 20           # EWMA halflife for realized vol
    exit_turnover_mult: float = 5.0  # Exit turnover = turnover_pct * this (faster exits)
    dd_throttle_enabled: bool = False
    dd_throttle_threshold: float = 100_000.0  # $ drawdown to trigger throttle
    dd_throttle_factor: float = 0.5  # Reduce entry turnover by this when in DD
    dd_riskoff_enabled: bool = False
    dd_riskoff_start: float = 100_000.0
    dd_riskoff_full: float = 400_000.0
    dd_riskoff_min_scale: float = 0.25
    cancel_cooldown_s: float = 0.25
    unknown_order_ttl_s: float = 2.0
    enabled: bool = True


@dataclass
class StrategyParams:
    """All strategy parameters loaded from config."""
    pair_coint: list[PairCointParams]
    etf_nav: EtfNavParams | None
    width: dict[str, float]  # Bid-ask spreads per ticker
    allocator: AllocatorConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StrategyParams':
        pair_coint = []
        etf_nav = None

        for s in data.get('strategies', []):
            if s.get('type') == 'pair_coint':
                pair_coint.append(PairCointParams(
                    a=s['a'],
                    b=s['b'],
                    c=s['c'],
                    beta=s['beta'],
                    std=s['std'],
                    pyramid=_parse_pyramid(s['pyramid']),
                    enabled=s.get('enabled', True),
                ))
            elif s.get('type') == 'etf_nav':
                etf_nav = EtfNavParams(
                    pyramid=_parse_pyramid(s['pyramid']),
                    eod_flat=s.get('eod_flat', False),
                    enabled=s.get('enabled', True),
                )

        # Parse allocator config
        allocator = None
        if 'allocator' in data:
            a = data['allocator']
            allocator = AllocatorConfig(
                gross_limit=a.get('gross_limit', 50_000_000.0),
                net_limit=a.get('net_limit', 10_000_000.0),
                max_shares=a.get('max_shares'),
                turnover_pct=a.get('turnover_pct', 0.10),
                min_threshold=a.get('min_threshold', 0.0),
                top_n=a.get('top_n', 4),
                horizon_bars=a.get('horizon_bars', 10),
                switch_lambda=a.get('switch_lambda', 0.10),
                regime_cutoff=a.get('regime_cutoff', 2.5),
                w_max=a.get('w_max', 1.0),
                vol_scale_enabled=a.get('vol_scale_enabled', False),
                target_vol=a.get('target_vol', 50_000.0),
                vol_halflife=a.get('vol_halflife', 20),
                exit_turnover_mult=a.get('exit_turnover_mult', 5.0),
                dd_throttle_enabled=a.get('dd_throttle_enabled', False),
                dd_throttle_threshold=a.get('dd_throttle_threshold', 100_000.0),
                dd_throttle_factor=a.get('dd_throttle_factor', 0.5),
                dd_riskoff_enabled=a.get('dd_riskoff_enabled', False),
                dd_riskoff_start=a.get('dd_riskoff_start', 100_000.0),
                dd_riskoff_full=a.get('dd_riskoff_full', 400_000.0),
                dd_riskoff_min_scale=a.get('dd_riskoff_min_scale', 0.25),
                cancel_cooldown_s=a.get('cancel_cooldown_s', 0.25),
                unknown_order_ttl_s=a.get('unknown_order_ttl_s', 2.0),
                enabled=a.get('enabled', True),
            )

        return cls(
            pair_coint=pair_coint,
            etf_nav=etf_nav,
            width=data.get('width', {}),
            allocator=allocator,
        )

    @classmethod
    def load(cls, path: str | Path) -> 'StrategyParams':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Default params path
DEFAULT_PARAMS_PATH = Path(__file__).parent / 'strategy_params.json'
