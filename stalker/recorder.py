import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi


@dataclass
class Recorder:
    """Records all available market data for offline replay."""

    client: RotmanInteractiveTraderApi
    output_dir: Path = field(default_factory=lambda: Path('data'))
    poll_interval: float = 0.25

    # Internal state
    _session_dir: Path = field(init=False, default=None)
    _files: dict[str, Any] = field(init=False, default_factory=dict)
    _tickers: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def _init_session(self, case: dict) -> None:
        """Create session directory and open file handles."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        case_name = case.get('name', 'unknown').replace(' ', '_')
        self._session_dir = self.output_dir / f'{timestamp}_{case_name}'
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Write case metadata
        with open(self._session_dir / 'meta.json', 'w') as f:
            json.dump({
                'case': case,
                'recorded_at': timestamp,
                'poll_interval': self.poll_interval,
            }, f, indent=2)

        # Open streaming files
        self._files = {
            'ticks': open(self._session_dir / 'ticks.jsonl', 'w'),
            'books': open(self._session_dir / 'books.jsonl', 'w'),
            'tas': open(self._session_dir / 'time_and_sales.jsonl', 'w'),
            'orders': open(self._session_dir / 'orders.jsonl', 'w'),
            'limits': open(self._session_dir / 'limits.jsonl', 'w'),
        }

        logging.info('Recording to %s', self._session_dir)

    def _close_files(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {}

    def _write(self, stream: str, data: dict) -> None:
        if stream in self._files:
            self._files[stream].write(json.dumps(data) + '\n')
            self._files[stream].flush()

    def _discover_tickers(self, portfolio: dict) -> list[str]:
        """Get all tradeable tickers from portfolio."""
        return [t for t in portfolio.keys() if portfolio[t].get('is_tradeable', True)]

    def _record_tick(self, case: dict, portfolio: dict) -> None:
        """Record one tick of data."""
        ts = time.time()
        period = case.get('period', 0)
        tick = case.get('tick', 0)

        # Securities snapshot
        self._write('ticks', {
            'ts': ts,
            'period': period,
            'tick': tick,
            'case_status': case.get('status'),
            'securities': portfolio,
        })

        # Order books for each ticker
        for ticker in self._tickers:
            try:
                book = self.client.get_order_book(ticker, limit=50)
                self._write('books', {
                    'ts': ts,
                    'period': period,
                    'tick': tick,
                    'ticker': ticker,
                    'book': book,
                })
            except Exception as e:
                logging.debug('Failed to get book for %s: %s', ticker, e)

        # Time and sales for each ticker
        for ticker in self._tickers:
            try:
                tas = self.client.get_time_and_sales(ticker)
                if tas:
                    self._write('tas', {
                        'ts': ts,
                        'period': period,
                        'tick': tick,
                        'ticker': ticker,
                        'trades': tas,
                    })
            except Exception as e:
                logging.debug('Failed to get T&S for %s: %s', ticker, e)

        # Current orders
        try:
            orders = self.client.get_orders()
            self._write('orders', {
                'ts': ts,
                'period': period,
                'tick': tick,
                'orders': orders,
            })
        except Exception as e:
            logging.debug('Failed to get orders: %s', e)

        # Limits
        try:
            limits = self.client.get_limits()
            trader = self.client.get_trader()
            self._write('limits', {
                'ts': ts,
                'period': period,
                'tick': tick,
                'limits': limits,
                'trader': trader,
            })
        except Exception as e:
            logging.debug('Failed to get limits: %s', e)

    def _record_history(self) -> None:
        """Record full price history at end of session."""
        history = {}
        for ticker in self._tickers:
            try:
                history[ticker] = self.client.get_history(ticker)
            except Exception as e:
                logging.debug('Failed to get history for %s: %s', ticker, e)

        with open(self._session_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def run(self) -> None:
        """Main recording loop."""
        logging.info('Waiting for market to open...')

        # Wait for active case
        while True:
            case = self.client.get_case()
            if case.get('status') == 'ACTIVE':
                break
            time.sleep(1)

        # Get initial portfolio to discover tickers
        portfolio = self.client.get_portfolio()
        self._tickers = self._discover_tickers(portfolio)
        logging.info('Tracking tickers: %s', self._tickers)

        # Initialize session
        self._init_session(case)

        last_period = case.get('period', 0)
        tick_count = 0

        try:
            while True:
                case = self.client.get_case()
                status = case.get('status')

                if status != 'ACTIVE':
                    if status == 'PAUSED':
                        logging.info('Market paused, waiting...')
                        time.sleep(1)
                        continue
                    else:
                        logging.info('Market stopped.')
                        break

                # Check for period change
                current_period = case.get('period', 0)
                if current_period != last_period:
                    logging.info('Period changed: %d -> %d', last_period, current_period)
                    last_period = current_period

                portfolio = self.client.get_portfolio()
                self._record_tick(case, portfolio)

                tick_count += 1
                if tick_count % 100 == 0:
                    logging.info('Recorded %d ticks (period=%d, tick=%d)',
                                 tick_count, case.get('period'), case.get('tick'))

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logging.info('Interrupted by user.')

        finally:
            logging.info('Recording complete. Total ticks: %d', tick_count)
            self._record_history()
            self._close_files()
            logging.info('Data saved to %s', self._session_dir)
