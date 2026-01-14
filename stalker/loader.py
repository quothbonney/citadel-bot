import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator


@dataclass
class Tick:
    """A single recorded tick."""
    ts: float
    period: int
    tick: int
    securities: dict


@dataclass
class BookSnapshot:
    """Order book snapshot for a ticker."""
    ts: float
    period: int
    tick: int
    ticker: str
    bids: list[dict]
    asks: list[dict]


@dataclass
class Trade:
    """A single trade from time & sales."""
    ts: float
    period: int
    tick: int
    ticker: str
    trade_id: int
    price: float
    quantity: float


class SessionLoader:
    """Load and replay recorded session data."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self._meta = None

    @property
    def meta(self) -> dict:
        if self._meta is None:
            with open(self.session_dir / 'meta.json') as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def case(self) -> dict:
        return self.meta.get('case', {})

    @property
    def tickers(self) -> list[str]:
        """Get tickers from first tick."""
        for tick in self.ticks():
            return list(tick.securities.keys())
        return []

    def ticks(self) -> Generator[Tick, None, None]:
        """Iterate through all tick snapshots."""
        path = self.session_dir / 'ticks.jsonl'
        if not path.exists():
            return

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                yield Tick(
                    ts=data['ts'],
                    period=data['period'],
                    tick=data['tick'],
                    securities=data['securities'],
                )

    def books(self, ticker: str = None) -> Generator[BookSnapshot, None, None]:
        """Iterate through order book snapshots, optionally filtered by ticker."""
        path = self.session_dir / 'books.jsonl'
        if not path.exists():
            return

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if ticker and data['ticker'] != ticker:
                    continue
                book = data.get('book', {})
                yield BookSnapshot(
                    ts=data['ts'],
                    period=data['period'],
                    tick=data['tick'],
                    ticker=data['ticker'],
                    bids=book.get('bid', []),
                    asks=book.get('ask', []),
                )

    def trades(self, ticker: str = None) -> Generator[Trade, None, None]:
        """Iterate through time & sales, optionally filtered by ticker."""
        path = self.session_dir / 'time_and_sales.jsonl'
        if not path.exists():
            return

        seen_ids: set[tuple[str, int]] = set()

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if ticker and data['ticker'] != ticker:
                    continue
                for t in data.get('trades', []):
                    key = (data['ticker'], t['id'])
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    yield Trade(
                        ts=data['ts'],
                        period=data['period'],
                        tick=data['tick'],
                        ticker=data['ticker'],
                        trade_id=t['id'],
                        price=t['price'],
                        quantity=t['quantity'],
                    )

    def history(self) -> dict[str, list[dict]]:
        """Load full OHLC history for all tickers."""
        path = self.session_dir / 'history.json'
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def limits(self) -> Generator[dict, None, None]:
        """Iterate through limit snapshots."""
        path = self.session_dir / 'limits.jsonl'
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                yield json.loads(line)


def list_sessions(data_dir: str | Path = 'data') -> list[Path]:
    """List all recorded sessions."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / 'meta.json').exists()])
