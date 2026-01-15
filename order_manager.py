from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
import logging

from RotmanInteractiveTraderApi import OrderAction, OrderStatus, OrderType, RotmanInteractiveTraderApi


class OrderRejectedByRiskLimit(RuntimeError):
    def __init__(self, *, ticker: str, side: str, qty: int, price: float | None, resp: object) -> None:
        super().__init__(
            f"order rejected by risk limit: ticker={ticker} side={side} qty={qty} price={price} resp={resp!r}"
        )
        self.ticker = ticker
        self.side = side
        self.qty = qty
        self.price = price
        self.resp = resp


class LocalState(str, Enum):
    SENT = "SENT"          # submitted locally
    LIVE = "LIVE"          # observed on server as OPEN
    CANCEL_SENT = "CANCEL_SENT"
    DONE = "DONE"          # observed CANCELLED or TRANSACTED


@dataclass
class TrackedOrder:
    order_id: int
    ticker: str
    side: OrderAction
    quantity: int
    price: float | None
    t_created: float
    t_cancel_sent: float | None = None
    state: LocalState = LocalState.SENT
    quantity_filled: float = 0.0
    status: OrderStatus | None = None

    @property
    def remaining(self) -> float:
        return max(0.0, float(self.quantity) - float(self.quantity_filled))

    @property
    def signed_remaining(self) -> float:
        sign = 1.0 if self.side == OrderAction.BUY else -1.0
        return sign * self.remaining


class OrderManager:
    """Minimal order lifecycle manager.

    Key rule: never run both BUY and SELL for the same ticker concurrently.
    If we need to flip, we cancel first and *wait* for confirmation via refresh().
    This avoids 'cancel didn't actually cancel' simulator gotchas turning into double fills.
    """

    def __init__(
        self,
        client: RotmanInteractiveTraderApi,
        *,
        cancel_cooldown_s: float = 0.25,
        unknown_order_ttl_s: float = 2.0,
    ) -> None:
        self.client = client
        self._orders: dict[int, TrackedOrder] = {}
        self.cancel_cooldown_s = float(cancel_cooldown_s)
        self.unknown_order_ttl_s = float(unknown_order_ttl_s)

    def refresh(self) -> None:
        """Reconcile local tracked orders with server state."""
        open_orders = {o["order_id"]: o for o in self.client.get_orders(OrderStatus.OPEN)}
        cancelled = {o["order_id"]: o for o in self.client.get_orders(OrderStatus.CANCELLED)}
        transacted = {o["order_id"]: o for o in self.client.get_orders(OrderStatus.TRANSACTED)}
        now = time.time()

        for oid, tr in list(self._orders.items()):
            if oid in open_orders:
                o = open_orders[oid]
                tr.status = OrderStatus.OPEN
                tr.quantity_filled = float(o.get("quantity_filled", 0.0))
                tr.state = LocalState.LIVE
                continue

            if oid in cancelled:
                o = cancelled[oid]
                tr.status = OrderStatus.CANCELLED
                tr.quantity_filled = float(o.get("quantity_filled", tr.quantity_filled))
                tr.state = LocalState.DONE
                continue

            if oid in transacted:
                o = transacted[oid]
                tr.status = OrderStatus.TRANSACTED
                tr.quantity_filled = float(o.get("quantity_filled", tr.quantity_filled))
                tr.state = LocalState.DONE
                continue

            # Unknown to server queries. Keep state as-is; do not assume cancellation worked.
            # But do not keep zombies forever.
            if (now - tr.t_created) > self.unknown_order_ttl_s:
                del self._orders[oid]

        # Drop DONE orders to keep memory bounded
        for oid in [oid for oid, tr in self._orders.items() if tr.state == LocalState.DONE]:
            del self._orders[oid]

    def effective_position_adjustments(self) -> dict[str, float]:
        """Conservative position adjustments from outstanding OPEN orders."""
        adj: dict[str, float] = {}
        for tr in self._orders.values():
            if tr.state in (LocalState.SENT, LocalState.LIVE, LocalState.CANCEL_SENT):
                adj[tr.ticker] = adj.get(tr.ticker, 0.0) + tr.signed_remaining
        return adj

    def open_orders_for_ticker(self, ticker: str) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.ticker == ticker]

    def cancel_ticker(self, ticker: str) -> None:
        ids = [o.order_id for o in self.open_orders_for_ticker(ticker)]
        if not ids:
            return
        self.client.cancel_orders(ids)
        now = time.time()
        for oid in ids:
            tr = self._orders.get(oid)
            if tr is not None and tr.state != LocalState.DONE:
                tr.state = LocalState.CANCEL_SENT
                tr.t_cancel_sent = now

    def submit(self, ticker: str, side: OrderAction, quantity: int, price: float | None) -> int:
        resp = self.client.place_order(
            ticker,
            OrderType.LIMIT,
            quantity,
            side,
            price=price,
        )
        if not isinstance(resp, dict) or "order_id" not in resp:
            # Handle known RIT error: risk limit rejection. Do not crash the whole bot for this.
            if isinstance(resp, dict):
                msg = str(resp.get("message") or "")
                if "exceed gross trading limits" in msg.lower():
                    raise OrderRejectedByRiskLimit(
                        ticker=ticker,
                        side=side.value,
                        qty=quantity,
                        price=price,
                        resp=resp,
                    )

            # Fail loudly with the actual server payload; do not guess.
            logging.error(
                "place_order unexpected response: ticker=%s side=%s qty=%s price=%s resp=%r",
                ticker,
                side.value,
                quantity,
                price,
                resp,
            )
            raise ValueError(
                f"place_order returned unexpected response (missing order_id): "
                f"ticker={ticker} side={side.value} qty={quantity} price={price} resp={resp!r}"
            )
        oid = int(resp["order_id"])
        tr = TrackedOrder(
            order_id=oid,
            ticker=ticker,
            side=side,
            quantity=int(quantity),
            price=float(price) if price is not None else None,
            t_created=time.time(),
            state=LocalState.SENT,
            quantity_filled=float(resp.get("quantity_filled", 0.0)),
            status=OrderStatus(resp.get("status", OrderStatus.OPEN.value)),
        )
        self._orders[oid] = tr
        return oid

    def reconcile_target_orders(self, desired: dict[str, tuple[OrderAction, int, float | None]]) -> None:
        """Cancel/submit to move toward desired per-ticker orders.

        desired maps ticker -> (side, qty, price). If ticker absent, we cancel any live orders.
        """
        # Cancel tickers not desired anymore
        for tr in list(self._orders.values()):
            if tr.ticker not in desired:
                self.cancel_ticker(tr.ticker)

        for ticker, (side, qty, price) in desired.items():
            existing = self.open_orders_for_ticker(ticker)
            if existing:
                # If we already have a same-side order, keep it (minimal).
                # If side differs, cancel and wait; do NOT place opposite in same tick.
                if any(o.side != side for o in existing):
                    self.cancel_ticker(ticker)
                continue

            # Cooldown after cancel: if we recently cancelled anything for this ticker,
            # do not place the opposite-side order immediately. This matches the probe's
            # "open_delay=NA but TRANSACTED after cancel" behavior.
            now = time.time()
            recently_cancelled = any(
                (o.t_cancel_sent is not None and (now - o.t_cancel_sent) < self.cancel_cooldown_s)
                for o in self._orders.values()
                if o.ticker == ticker
            )
            if recently_cancelled:
                continue

            if qty <= 0:
                continue
            try:
                self.submit(ticker, side, qty, price)
            except OrderRejectedByRiskLimit as e:
                logging.warning("%s", e)
                # Stop submitting further orders this tick; we're at the server's limit.
                return


