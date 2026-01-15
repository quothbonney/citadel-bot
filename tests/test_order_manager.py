from __future__ import annotations

from dataclasses import dataclass

import pytest

from RotmanInteractiveTraderApi import OrderAction, OrderStatus
from order_manager import OrderManager, OrderRejectedByRiskLimit


@dataclass
class _FakeOrder:
    order_id: int
    ticker: str
    action: str
    quantity: int
    quantity_filled: float = 0.0
    status: str = OrderStatus.OPEN.value
    price: float | None = None


class FakeClient:
    """Fake RIT client to simulate 'cancel didn't prevent later fill' behavior."""

    def __init__(self) -> None:
        self._next_id = 1
        self._orders: dict[int, _FakeOrder] = {}

    def place_order(self, ticker, order_type, quantity, action, price=None, dry_run=None):
        oid = self._next_id
        self._next_id += 1
        o = _FakeOrder(order_id=oid, ticker=ticker, action=action.value, quantity=quantity, price=price)
        self._orders[oid] = o
        return {
            "order_id": oid,
            "ticker": ticker,
            "quantity": quantity,
            "action": action.value,
            "price": price,
            "quantity_filled": 0.0,
            "status": OrderStatus.OPEN.value,
        }

    def cancel_orders(self, order_ids):
        # Gotcha: acknowledge cancel, but do NOT change server state.
        return {"cancelled_order_ids": list(order_ids)}

    def get_orders(self, status=OrderStatus.OPEN):
        out = []
        for o in self._orders.values():
            if o.status == status.value:
                out.append(
                    {
                        "order_id": o.order_id,
                        "ticker": o.ticker,
                        "quantity": o.quantity,
                        "action": o.action,
                        "price": o.price,
                        "quantity_filled": o.quantity_filled,
                        "status": o.status,
                    }
                )
        return out

    def force_fill(self, order_id: int) -> None:
        o = self._orders[order_id]
        o.quantity_filled = float(o.quantity)
        o.status = OrderStatus.TRANSACTED.value


class FakeClientNoOrderId(FakeClient):
    def place_order(self, ticker, order_type, quantity, action, price=None, dry_run=None):
        # Simulate API rejection/rate limit payload that lacks 'order_id'
        return {"error": "rejected", "detail": "simulated"}


class FakeClientRiskLimit(FakeClient):
    def place_order(self, ticker, order_type, quantity, action, price=None, dry_run=None):
        return {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Order request cannot be submitted because it will exceed gross trading limits in the risk category LIMIT-STOCK.",
        }


class FakeClientRiskLimitNet(FakeClient):
    def place_order(self, ticker, order_type, quantity, action, price=None, dry_run=None):
        return {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Order request cannot be submitted because it will exceed net trading limits in the risk category LIMIT-STOCK.",
        }


def test_order_manager_wont_flip_sides_same_tick_when_cancel_is_unreliable():
    c = FakeClient()
    om = OrderManager(c, cancel_cooldown_s=0.25, unknown_order_ttl_s=10.0)  # type: ignore[arg-type]

    # Place a BUY
    desired = {"AAA": (OrderAction.BUY, 100, 10.0)}
    om.reconcile_target_orders(desired)
    assert len(c.get_orders(OrderStatus.OPEN)) == 1

    # Now desire a SELL instead. Manager should cancel but NOT place opposite in same tick.
    desired2 = {"AAA": (OrderAction.SELL, 100, 9.9)}
    om.reconcile_target_orders(desired2)
    # Still only the original OPEN order exists, no new SELL.
    open_orders = c.get_orders(OrderStatus.OPEN)
    assert len(open_orders) == 1
    assert open_orders[0]["action"] == OrderAction.BUY.value

    # Late fill happens even though we "cancelled"
    oid = open_orders[0]["order_id"]
    c.force_fill(oid)
    om.refresh()

    # After refresh, the old order is DONE and can be replaced on next reconcile
    om.reconcile_target_orders(desired2)
    open_orders2 = c.get_orders(OrderStatus.OPEN)
    assert len(open_orders2) == 1
    assert open_orders2[0]["action"] == OrderAction.SELL.value


def test_order_manager_cancel_cooldown_blocks_immediate_resubmit_after_cancel():
    c = FakeClient()
    om = OrderManager(c, cancel_cooldown_s=10.0, unknown_order_ttl_s=10.0)  # long cooldown

    # Place BUY then request SELL to trigger cancel_sent timestamp
    om.reconcile_target_orders({"AAA": (OrderAction.BUY, 100, 10.0)})
    om.reconcile_target_orders({"AAA": (OrderAction.SELL, 100, 9.9)})

    # Order is still OPEN, and cooldown should prevent any new order submission.
    assert len(c.get_orders(OrderStatus.OPEN)) == 1


def test_order_manager_raises_clear_error_if_place_order_returns_no_order_id():
    c = FakeClientNoOrderId()
    om = OrderManager(c, cancel_cooldown_s=0.25, unknown_order_ttl_s=10.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError) as e:
        om.reconcile_target_orders({"AAA": (OrderAction.BUY, 100, 10.0)})

    assert "missing order_id" in str(e.value)


def test_order_manager_handles_risk_limit_rejection_without_crashing():
    c = FakeClientRiskLimit()
    om = OrderManager(c, cancel_cooldown_s=0.0, unknown_order_ttl_s=10.0)  # type: ignore[arg-type]

    # reconcile_target_orders should swallow ONLY OrderRejectedByRiskLimit and return
    om.reconcile_target_orders({"AAA": (OrderAction.BUY, 100, 10.0)})

    # submit() should still raise directly if called
    with pytest.raises(OrderRejectedByRiskLimit):
        om.submit("AAA", OrderAction.BUY, 100, 10.0)


def test_order_manager_handles_net_risk_limit_rejection_without_crashing():
    c = FakeClientRiskLimitNet()
    om = OrderManager(c, cancel_cooldown_s=0.0, unknown_order_ttl_s=10.0)  # type: ignore[arg-type]

    om.reconcile_target_orders({"AAA": (OrderAction.BUY, 100, 10.0)})

    with pytest.raises(OrderRejectedByRiskLimit):
        om.submit("AAA", OrderAction.BUY, 100, 10.0)


