from abc import ABC, abstractmethod

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi
from market import Market, market


class Strategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, client: RotmanInteractiveTraderApi, mkt: Market = market) -> None:
        self.client = client
        self.market = mkt

    def get_prices(self, portfolio: dict) -> dict[str, float]:
        """Extract mid prices from portfolio."""
        prices = {}
        for ticker in self.market.all_tickers:
            sec = portfolio.get(ticker)
            if sec:
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                prices[ticker] = (bid + ask) / 2 if bid and ask else sec.get('last', 0)
        return prices

    def get_positions(self, portfolio: dict) -> dict[str, float]:
        """Extract positions from portfolio."""
        return {t: portfolio.get(t, {}).get('position', 0) for t in self.market.all_tickers}

    @abstractmethod
    def on_tick(self, portfolio: dict, case: dict) -> None:
        """Called each tick. Implement your strategy logic here."""
        pass


class ExampleStrategy(Strategy):
    """Example strategy showing market structure usage."""

    def on_tick(self, portfolio: dict, case: dict) -> None:
        prices = self.get_prices(portfolio)
        positions = self.get_positions(portfolio)

        # Calculate ETF NAV
        nav = self.market.nav(prices)
        etf_price = prices.get(self.market.etf, 0)

        # Check risk limits before trading
        ok, gross, net = self.market.check_limits(positions, prices)
        if not ok:
            return  # Over limits, don't trade

        # Example: ETF vs NAV spread
        # if etf_price > nav:  # ETF overvalued
        #     sell ETF, buy stocks
        # if etf_price < nav:  # ETF undervalued
        #     buy ETF, sell stocks

        pass
