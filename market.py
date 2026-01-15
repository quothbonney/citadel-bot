from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """Invariant market structure (tickers only - limits in AllocatorConfig)."""

    # Underlying stocks
    stocks: tuple[str, ...] = ('AAA', 'BBB', 'CCC', 'DDD')

    # ETF that holds equal shares of all stocks
    # NAV = average price of underlying stocks
    etf: str = 'ETF'

    # Index ETF tracking top 100 stocks (includes our stocks)
    index: str = 'IND'

    @property
    def all_tickers(self) -> tuple[str, ...]:
        return self.stocks + (self.etf, self.index)

    def nav(self, prices: dict[str, float]) -> float:
        """Calculate ETF NAV as average of underlying stock prices."""
        return sum(prices[s] for s in self.stocks) / len(self.stocks)

    def gross_exposure(self, positions: dict[str, float], prices: dict[str, float]) -> float:
        """Sum of absolute position values."""
        return sum(abs(positions.get(t, 0) * prices.get(t, 0)) for t in self.all_tickers)

    def net_exposure(self, positions: dict[str, float], prices: dict[str, float]) -> float:
        """Sum of position values (longs and shorts net out)."""
        return sum(positions.get(t, 0) * prices.get(t, 0) for t in self.all_tickers)

    def check_limits(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        gross_limit: float = 50_000_000.0,
        net_limit: float = 10_000_000.0,
    ) -> tuple[bool, float, float]:
        """Check if within risk limits. Returns (ok, gross, net)."""
        gross = self.gross_exposure(positions, prices)
        net = self.net_exposure(positions, prices)
        ok = gross <= gross_limit and abs(net) <= net_limit
        return ok, gross, net


# Default market instance
market = Market()
