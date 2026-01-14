import json
from typing import Any, Optional, TypedDict

import requests
from enum import Enum
import logging


class OrderAction(str, Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderType(str, Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'


class OrderStatus(str, Enum):
    OPEN = 'OPEN'
    TRANSACTED = 'TRANSACTED'
    CANCELLED = 'CANCELLED'


class CaseStatus(str, Enum):
    ACTIVE = 'ACTIVE'
    PAUSED = 'PAUSED'
    STOPPED = 'STOPPED'


class SecurityType(str, Enum):
    SPOT = 'SPOT'
    FUTURE = 'FUTURE'
    INDEX = 'INDEX'
    OPTION = 'OPTION'
    STOCK = 'STOCK'
    CURRENCY = 'CURRENCY'
    BOND = 'BOND'
    RATE = 'RATE'
    FORWARD = 'FORWARD'
    SWAP = 'SWAP'
    SWAP_BOM = 'SWAP_BOM'
    SPRE = 'SPRE'


class AssetType(str, Enum):
    CONTAINER = 'CONTAINER'
    PIPELINE = 'PIPELINE'
    SHIP = 'SHIP'
    REFINERY = 'REFINERY'
    POWER_PLANT = 'POWER_PLANT'
    PRODUCER = 'PRODUCER'


# TypedDict definitions for API responses

class TickerQuantity(TypedDict):
    ticker: str
    quantity: float


class TickerPrice(TypedDict):
    ticker: str
    price: float


class Case(TypedDict):
    name: str
    period: int
    tick: int
    ticks_per_period: int
    total_periods: int
    status: str
    is_enforce_trading_limits: bool


class Trader(TypedDict):
    trader_id: str
    first_name: str
    last_name: str
    nlv: float


class Limit(TypedDict):
    name: str
    gross: float
    net: float
    gross_limit: float
    net_limit: float
    gross_fine: float
    net_fine: float


class Order(TypedDict):
    order_id: int
    period: int
    tick: int
    trader_id: str
    ticker: str
    type: str
    quantity: float
    action: str
    price: Optional[float]
    quantity_filled: float
    vwap: Optional[float]
    status: str


class SecurityLimit(TypedDict):
    name: str
    units: float


class Security(TypedDict):
    ticker: str
    type: str
    size: int
    position: float
    vwap: float
    nlv: float
    last: float
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    volume: float
    unrealized: float
    realized: float
    currency: str
    total_volume: float
    limits: list[SecurityLimit]
    interest_rate: float
    is_tradeable: bool
    is_shortable: bool
    start_period: int
    stop_period: int
    description: str
    unit_multiplier: int
    display_unit: str
    start_price: float
    min_price: float
    max_price: float
    quoted_decimals: int
    trading_fee: float
    limit_order_rebate: float
    min_trade_size: int
    max_trade_size: int
    required_tickers: Optional[list[str]]
    underlying_tickers: Optional[list[str]]
    bond_coupon: float
    interest_payments_per_period: int
    base_security: str
    fixing_ticker: Optional[str]
    api_orders_per_second: int
    execution_delay_ms: int
    interest_rate_ticker: Optional[str]
    otc_price_range: float


class OrderBook(TypedDict):
    bid: list[Order]
    ask: list[Order]


class TimeAndSalesEntry(TypedDict):
    id: int
    period: int
    tick: int
    price: float
    quantity: float


class HistoryEntry(TypedDict):
    tick: int
    open: float
    high: float
    low: float
    close: float


class Asset(TypedDict):
    ticker: str
    type: str
    description: str
    total_quantity: int
    available_quantity: int
    lease_price: float
    convert_from: list[TickerQuantity]
    convert_to: list[TickerQuantity]
    containment: int
    ticks_per_conversion: int
    ticks_per_lease: int
    is_available: bool
    start_period: int
    stop_period: int


class AssetLease(TypedDict):
    id: int
    ticker: str
    type: str
    start_lease_period: int
    start_lease_tick: int
    next_lease_period: int
    next_lease_tick: int
    convert_from: list[TickerQuantity]
    convert_to: list[TickerQuantity]
    containment_usage: int


class CancelResponse(TypedDict):
    cancelled_order_ids: list[int]


class SuccessResponse(TypedDict):
    success: bool


class RotmanInteractiveTraderApi:
    """
    Partial implementation of https://rit.306w.ca/RIT-REST-API-DEV/1.0.3/.
    """

    def __init__(self, api_key: str, api_host: str = 'http://localhost:9999') -> None:
        self.api_key = api_key
        self.api_host = api_host
        self.api_version = 'v1'

    def make_request(self, method: str, endpoint: str, params: Optional[dict[str, Any]] = None) -> Any:
        req = requests.Request(
            method=method,
            url=f'{self.api_host}/{self.api_version}/{endpoint}',
            headers={
                'X-API-Key': self.api_key,
                'Accept': 'application/json'
            },
            params=params
        )
        p = req.prepare()
        s = requests.Session()
        logging.debug(f'{p.method} {p.url}')
        r = s.send(p).json()
        if not p.method == 'GET':
            logging.debug(f'{p.method} {p.url} {json.dumps(r, indent=2)}')
        return r

    def get_case(self) -> Case:
        return self.make_request('get', 'case')

    def is_market_open(self) -> bool:
        return self.get_case()['status'] == 'ACTIVE'

    def get_orders(self, status: OrderStatus = OrderStatus.OPEN) -> list[Order]:
        return self.make_request('get', 'orders', {
            'status': status.value
        })

    def get_time_and_sales(self, ticker: str) -> list[TimeAndSalesEntry]:
        return self.make_request('get', 'securities/tas', {
            'ticker': ticker
        })

    def get_history(self, ticker: str) -> list[HistoryEntry]:
        return self.make_request('get', 'securities/history', {
            'ticker': ticker
        })

    def get_trader(self) -> Trader:
        return self.make_request('get', 'trader')

    def get_limits(self) -> dict[str, Limit]:
        limits: dict[str, Limit] = {}
        for item in self.make_request('get', 'limits'):
            limits[item['name']] = item
        return limits

    def get_portfolio(self) -> dict[str, Security]:
        portfolio: dict[str, Security] = {}
        for item in self.make_request('get', 'securities'):
            portfolio[item['ticker']] = item
        return portfolio

    def get_order_book(self, ticker: str, limit: int = 20) -> OrderBook:
        return self.make_request('get', 'securities/book', {
            'ticker': ticker,
            'limit': limit
        })

    def get_order_fills(self) -> list[Order]:
        partial = list(filter(lambda order: order['quantity_filled'] > 0, self.get_orders(OrderStatus.OPEN)))
        transacted = self.get_orders(OrderStatus.TRANSACTED)
        return partial + transacted

    def cancel_all_orders(self, ticker: Optional[str] = None) -> CancelResponse:
        return self.make_request('post', 'commands/cancel', {
            'ticker': ticker,
            'all': 1 if ticker is None else None
        })

    def place_order(self, ticker: str, order_type: OrderType, quantity: int, action: OrderAction, price: Optional[float] = None, dry_run: Optional[bool] = None) -> Order:
        if not self.is_market_open():
            raise Exception('Market is closed.')
        return self.make_request('post', 'orders', params={
            'ticker': ticker,
            'type': order_type.value,
            'quantity': quantity,
            'action': action.value,
            'price': price,
            'dry_run': 1 if dry_run else None
        })

    def cancel_orders(self, order_ids: list[int]) -> CancelResponse:
        return self.make_request('post', f'commands/cancel', {
            'ids': ','.join(map(str, order_ids))
        })

    def get_assets(self) -> dict[str, Asset]:
        assets: dict[str, Asset] = {}
        for item in self.make_request('get', 'assets'):
            assets[item['ticker']] = item
        return assets

    def get_leases(self) -> list[AssetLease]:
        return self.make_request('get', 'leases')

    def lease_asset(self, ticker: str) -> AssetLease:
        return self.make_request('post', 'leases', {
            'ticker': ticker
        })

    def unlease_asset(self, lease_id: int) -> SuccessResponse:
        return self.make_request('delete', f'leases/{lease_id}')

    def use_lease(self, lease_id: int, convert_from: dict[str, int]) -> AssetLease:
        params: dict[str, Any] = {}
        i = 1
        for ticker in convert_from:
            params[f'from{i}'] = ticker
            params[f'quantity{i}'] = convert_from[ticker]
            i += 1
        return self.make_request('post', f'leases/{lease_id}', params)
