#!/usr/bin/env python3
"""
Microstructure probe for the RIT simulator.

Goal: empirically measure simulator-specific behavior like:
- Order visibility latency: place_order -> appears in get_orders(OPEN)
- Cancel semantics: cancel ack -> does it still fill later?
- Any batching cadence (e.g. ~0.2s execution queue) via timing clustering

This is intentionally small and non-invasive: it does NOT try to trade profitably.

Usage:
  python microstructure_probe.py --api-key ... --api-host http://localhost:9999 --ticker IND

Notes:
  - This uses only endpoints already present in RotmanInteractiveTraderApi.py:
      place_order, cancel_orders, get_orders
  - It will place and cancel real orders. Use small qty.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

from RotmanInteractiveTraderApi import OrderAction, OrderStatus, OrderType, RotmanInteractiveTraderApi


@dataclass(frozen=True)
class Sample:
    order_id: int
    t_place: float
    t_visible_open: float | None
    t_cancel_sent: float
    t_done: float | None
    final_status: str | None
    filled_after_cancel: bool


def _find_order(client: RotmanInteractiveTraderApi, order_id: int) -> dict | None:
    # Search across statuses for this order_id.
    for st in (OrderStatus.OPEN, OrderStatus.CANCELLED, OrderStatus.TRANSACTED):
        for o in client.get_orders(st):
            if int(o["order_id"]) == int(order_id):
                return o
    return None


def _poll_until(
    client: RotmanInteractiveTraderApi,
    order_id: int,
    timeout_s: float,
    poll_s: float,
) -> tuple[float | None, dict | None]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        o = _find_order(client, order_id)
        if o is not None:
            return time.time(), o
        time.sleep(poll_s)
    return None, None


def _poll_until_done(
    client: RotmanInteractiveTraderApi,
    order_id: int,
    timeout_s: float,
    poll_s: float,
) -> tuple[float | None, dict | None]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        o = _find_order(client, order_id)
        if o is not None:
            st = o.get("status")
            if st in (OrderStatus.CANCELLED.value, OrderStatus.TRANSACTED.value):
                return time.time(), o
        time.sleep(poll_s)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe RIT microstructure timing/cancel semantics")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--api-host", default="http://localhost:9999")
    ap.add_argument("--ticker", default="IND")
    ap.add_argument("--side", choices=("BUY", "SELL"), default="BUY")
    ap.add_argument("--qty", type=int, default=100)
    ap.add_argument("--price-offset", type=float, default=0.01, help="Offset from mid: BUY uses mid-offset, SELL uses mid+offset")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sleep-between", type=float, default=0.05, help="Seconds to sleep between iterations")
    ap.add_argument("--poll", type=float, default=0.01, help="Polling interval for order visibility/status")
    ap.add_argument("--visible-timeout", type=float, default=0.5, help="Timeout to wait for OPEN visibility")
    ap.add_argument("--done-timeout", type=float, default=2.0, help="Timeout to wait for CANCELLED/TRANSACTED")
    ap.add_argument("--cancel-delay", type=float, default=0.05, help="How long to wait after place before sending cancel")
    args = ap.parse_args()

    client = RotmanInteractiveTraderApi(args.api_key, args.api_host)
    case = client.get_case()
    if case.get("status") != "ACTIVE":
        raise RuntimeError(f"market not ACTIVE: {case}")

    portfolio = client.get_portfolio()
    sec = portfolio.get(args.ticker)
    if not sec:
        raise RuntimeError(f"ticker not found in portfolio: {args.ticker}")
    bid = float(sec.get("bid", 0) or 0)
    ask = float(sec.get("ask", 0) or 0)
    mid = (bid + ask) / 2 if bid and ask else float(sec.get("last", 0) or 0)
    if mid <= 0:
        raise RuntimeError(f"invalid mid for {args.ticker}: bid={bid} ask={ask} last={sec.get('last')}")

    side = OrderAction.BUY if args.side == "BUY" else OrderAction.SELL
    if side == OrderAction.BUY:
        price = mid - float(args.price_offset)
    else:
        price = mid + float(args.price_offset)

    print(f"Case: period={case.get('period')} tick={case.get('tick')} status={case.get('status')}")
    print(f"Probe: ticker={args.ticker} side={side.value} qty={args.qty} price={price:.4f} (mid={mid:.4f})")
    print(f"Iters={args.iters} cancel_delay={args.cancel_delay}s poll={args.poll}s")
    print()

    samples: list[Sample] = []

    for i in range(args.iters):
        t0 = time.time()
        resp = client.place_order(args.ticker, OrderType.LIMIT, args.qty, side, price=price)
        oid = int(resp["order_id"])

        t_open, o_open = _poll_until(client, oid, timeout_s=args.visible_timeout, poll_s=args.poll)

        time.sleep(args.cancel_delay)
        t_cancel = time.time()
        client.cancel_orders([oid])

        t_done, o_done = _poll_until_done(client, oid, timeout_s=args.done_timeout, poll_s=args.poll)
        final_status = o_done.get("status") if o_done is not None else None

        filled_after_cancel = False
        if o_done is not None:
            filled = float(o_done.get("quantity_filled", 0.0) or 0.0)
            if final_status == OrderStatus.TRANSACTED.value and filled > 0:
                # If it transacted after cancel was sent, that's the gotcha.
                filled_after_cancel = True

        samples.append(
            Sample(
                order_id=oid,
                t_place=t0,
                t_visible_open=t_open,
                t_cancel_sent=t_cancel,
                t_done=t_done,
                final_status=final_status,
                filled_after_cancel=filled_after_cancel,
            )
        )

        open_delay = (t_open - t0) if t_open is not None else None
        done_delay = (t_done - t_cancel) if t_done is not None else None
        print(
            f"{i+1:03d} oid={oid:6d} "
            f"open_delay={open_delay if open_delay is not None else 'NA'} "
            f"done_delay={done_delay if done_delay is not None else 'NA'} "
            f"final={final_status} late_fill={filled_after_cancel}"
        )

        time.sleep(args.sleep_between)

    print()

    open_delays = [s.t_visible_open - s.t_place for s in samples if s.t_visible_open is not None]
    done_delays = [s.t_done - s.t_cancel_sent for s in samples if s.t_done is not None]
    late_fills = sum(1 for s in samples if s.filled_after_cancel)

    def _summ(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"{name}: no samples")
            return
        print(
            f"{name}: n={len(xs)} "
            f"mean={statistics.mean(xs):.4f}s "
            f"p50={statistics.median(xs):.4f}s "
            f"p90={statistics.quantiles(xs, n=10)[8]:.4f}s "
            f"min={min(xs):.4f}s max={max(xs):.4f}s"
        )

    _summ("OPEN visibility delay", open_delays)
    _summ("Done-after-cancel delay", done_delays)
    print(f"Late fills after cancel: {late_fills}/{len(samples)}")

    # Rough batch-cadence hint: differences between successive OPEN-visible timestamps.
    if len(open_delays) >= 5:
        open_times = [s.t_visible_open for s in samples if s.t_visible_open is not None]
        if len(open_times) >= 5:
            diffs = [open_times[i] - open_times[i - 1] for i in range(1, len(open_times))]
            _summ("Δ between OPEN-visible events", diffs)


if __name__ == "__main__":
    main()


