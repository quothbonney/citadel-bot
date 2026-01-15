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
    turnover_k: float = 50_000.0
    min_threshold: float = 0.12  # Minimum |spread| to be considered
    top_n: int = 4  # Max signals to allocate to
    horizon_bars: int = 10
    switch_lambda: float = 0.10
    regime_cutoff: float = 2.5
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
                turnover_k=a.get('turnover_k', 50_000.0),
                min_threshold=a.get('min_threshold', 0.12),
                top_n=a.get('top_n', 4),
                horizon_bars=a.get('horizon_bars', 10),
                switch_lambda=a.get('switch_lambda', 0.10),
                regime_cutoff=a.get('regime_cutoff', 2.5),
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
