from collections import defaultdict

from params import EtfNavParams
from market import Market, market
from .base import SignalStrategy, Signal, Order, PositionState


class RunningMean:
    """Online computation of expanding mean."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n

    def get(self) -> float | None:
        return self.mean if self.n > 0 else None


class EtfNavStrategy(SignalStrategy):
    """
    ETF-NAV arbitrage strategy.

    Trades the spread: ETF - mean(AAA, BBB, CCC, DDD)
    Entry: |spread_adj| >= entry_abs
    Exit: spread crosses 0
    """

    def __init__(self, params: EtfNavParams, mkt: Market = market) -> None:
        super().__init__(params.strategy_id)
        self.params = params
        self.market = mkt

        # Seasonality: expanding mean of spread per timestep
        self._spread_means: dict[int, RunningMean] = defaultdict(RunningMean)
        self._spread_adj_cache: dict[tuple[int, int], float] = {}  # Cache per (period, tick)
        self._last_spread_adj: float | None = None

    def _get_mid(self, sec: dict) -> float:
        bid = sec.get('bid', 0)
        ask = sec.get('ask', 0)
        if bid and ask:
            return (bid + ask) / 2
        return sec.get('last', 0)

    def compute_signal(self, portfolio: dict, case: dict) -> Signal:
        etf = self.market.etf
        stocks = self.market.stocks

        sec_etf = portfolio.get(etf)
        if not sec_etf:
            return self.hold('missing ETF')

        stock_secs = {s: portfolio.get(s) for s in stocks}
        if not all(stock_secs.values()):
            return self.hold('missing stocks')

        etf_mid = self._get_mid(sec_etf)
        stock_mids = {s: self._get_mid(stock_secs[s]) for s in stocks}
        if etf_mid <= 0 or any(p <= 0 for p in stock_mids.values()):
            return self.hold('invalid prices')

        # NAV = average of stock prices
        nav = sum(stock_mids.values()) / len(stocks)
        spread = etf_mid - nav

        # Seasonality adjustment
        period = case.get('period', 0)
        tick = case.get('tick', 0)
        tick_key = (period, tick)
        spread_mean = self._spread_means[tick]

        # Only compute spread_adj once per (period, tick) to handle multiple snapshots per tick
        if tick_key in self._spread_adj_cache:
            spread_adj = self._spread_adj_cache[tick_key]
        else:
            prev_mean = spread_mean.get()
            spread_adj = spread - prev_mean if prev_mean is not None else spread
            self._spread_adj_cache[tick_key] = spread_adj
            spread_mean.update(spread)

        # State machine
        if self.state == PositionState.FLAT:
            if spread_adj >= self.params.entry_abs:
                # ETF overvalued: sell ETF, buy stocks
                self.state = PositionState.SHORT
                self._last_spread_adj = spread_adj
                orders = [Order(etf, 1, 'SELL', sec_etf.get('bid', etf_mid))]
                for s in stocks:
                    sec = stock_secs[s]
                    orders.append(Order(s, 1, 'BUY', sec.get('ask', stock_mids[s])))
                return self.enter_short(orders, f'spread_adj={spread_adj:.4f}')

            elif spread_adj <= -self.params.entry_abs:
                # ETF undervalued: buy ETF, sell stocks
                self.state = PositionState.LONG
                self._last_spread_adj = spread_adj
                orders = [Order(etf, 1, 'BUY', sec_etf.get('ask', etf_mid))]
                for s in stocks:
                    sec = stock_secs[s]
                    orders.append(Order(s, 1, 'SELL', sec.get('bid', stock_mids[s])))
                return self.enter_long(orders, f'spread_adj={spread_adj:.4f}')

        elif self.state == PositionState.LONG:
            # Exit when spread crosses 0
            if spread_adj >= 0:
                self.state = PositionState.FLAT
                orders = [Order(etf, 1, 'SELL', sec_etf.get('bid', etf_mid))]
                for s in stocks:
                    sec = stock_secs[s]
                    orders.append(Order(s, 1, 'BUY', sec.get('ask', stock_mids[s])))
                return self.exit(orders, f'spread crossed 0: {spread_adj:.4f}')

        elif self.state == PositionState.SHORT:
            # Exit when spread crosses 0
            if spread_adj <= 0:
                self.state = PositionState.FLAT
                orders = [Order(etf, 1, 'BUY', sec_etf.get('ask', etf_mid))]
                for s in stocks:
                    sec = stock_secs[s]
                    orders.append(Order(s, 1, 'SELL', sec.get('bid', stock_mids[s])))
                return self.exit(orders, f'spread crossed 0: {spread_adj:.4f}')

        self._last_spread_adj = spread_adj
        return self.hold(f'spread_adj={spread_adj:.4f}')
