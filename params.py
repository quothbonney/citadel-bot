import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairCointParams:
    """Parameters for a pair cointegration strategy."""
    a: str  # Long leg ticker
    b: str  # Hedge leg ticker
    c: float  # Intercept from regression
    beta: float  # Cointegration coefficient
    std: float  # Standard deviation of z
    entry_std: float | None = None  # Entry when |z| > entry_std * std (std mode)
    entry_abs: float | None = None  # Entry when |z * price_a| > entry_abs (dollar mode)
    enabled: bool = True

    @property
    def strategy_id(self) -> str:
        return f'pair_{self.a}_{self.b}'

    @property
    def use_std_mode(self) -> bool:
        """True if using std-based entry, False if using absolute dollar entry."""
        return self.entry_std is not None


@dataclass(frozen=True)
class EtfNavParams:
    """Parameters for ETF-NAV arbitrage strategy."""
    entry_abs: float  # Spread threshold for entry
    enabled: bool = True

    @property
    def strategy_id(self) -> str:
        return 'etf_nav'


@dataclass
class StrategyParams:
    """All strategy parameters loaded from config."""
    pair_coint: list[PairCointParams]
    etf_nav: EtfNavParams | None
    width: dict[str, float]  # Bid-ask spreads per ticker

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
                    entry_std=s.get('entry_std'),  # None if not specified
                    entry_abs=s.get('entry_abs'),  # None if not specified
                    enabled=s.get('enabled', True),
                ))
            elif s.get('type') == 'etf_nav':
                etf_nav = EtfNavParams(
                    entry_abs=s['entry_abs'],
                    enabled=s.get('enabled', True),
                )

        return cls(
            pair_coint=pair_coint,
            etf_nav=etf_nav,
            width=data.get('width', {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> 'StrategyParams':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Default params path
DEFAULT_PARAMS_PATH = Path(__file__).parent / 'strategy_params.json'
