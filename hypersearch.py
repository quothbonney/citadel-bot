#!/usr/bin/env python3
"""
Hyperparameter search for allocator params.

Usage:
    python hypersearch.py [session_dir]
    python hypersearch.py --samples 500  # random search with 500 samples
"""

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backtest import Backtest
from params import StrategyParams
from stalker import SessionLoader


@dataclass
class Result:
    params: dict
    net_pnl: float
    sharpe: float
    max_dd: float
    realized: float
    costs: float


def run_backtest(session: SessionLoader, base_params: dict, overrides: dict, scale: int = 1000) -> Result:
    """Run a single backtest with param overrides."""
    # Merge overrides into allocator config
    params_dict = json.loads(json.dumps(base_params))  # deep copy
    for k, v in overrides.items():
        params_dict["allocator"][k] = v

    params = StrategyParams.from_dict(params_dict)
    bt = Backtest(params, scale=scale)

    for tick in session.ticks():
        portfolio = tick.securities
        case = {"period": tick.period, "tick": tick.tick, "status": "ACTIVE"}
        bt.process_tick(portfolio, case)

    # Extract results
    tracker = bt.pnl.get("allocator")
    if tracker is None or not tracker.pnl_curve:
        return Result(params=overrides, net_pnl=0.0, sharpe=0.0, max_dd=0.0, realized=0.0, costs=0.0)

    net_pnl = tracker.pnl_curve[-1]
    realized = tracker.total_realized()
    costs = tracker.costs

    # Sharpe
    sharpe = 0.0
    if len(tracker.pnl_curve) > 10:
        returns = np.diff(tracker.pnl_curve)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390)

    # Max DD
    curve = np.array(tracker.pnl_curve)
    peak = np.maximum.accumulate(curve)
    max_dd = (curve - peak).min()

    return Result(
        params=overrides,
        net_pnl=net_pnl,
        sharpe=sharpe,
        max_dd=max_dd,
        realized=realized,
        costs=costs,
    )


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("session", nargs="?", help="Session directory")
    parser.add_argument("--scale", "-s", type=int, default=1000)
    parser.add_argument("--top", "-t", type=int, default=10, help="Show top N results")
    parser.add_argument("--samples", "-n", type=int, default=0, help="Random sample size (0=full grid)")
    args = parser.parse_args()

    # Find session
    if args.session:
        session_path = Path(args.session)
    else:
        downloads = Path("/Users/wudd/Downloads")
        data_dir = downloads / "data"
        if data_dir.exists():
            sessions = sorted([d for d in data_dir.iterdir() if d.is_dir() and "Citadel" in d.name])
        else:
            sessions = sorted([d for d in downloads.iterdir() if d.is_dir() and "Citadel" in d.name])
        if not sessions:
            print("No session found")
            sys.exit(1)
        session_path = sessions[-1]

    print(f"Session: {session_path.name}")
    session = SessionLoader(session_path)

    # Load base params
    base_path = Path(__file__).parent / "strategy_params.json"
    with open(base_path) as f:
        base_params = json.load(f)

    # Define search grid
    grid = {
        "turnover_pct": [0.005, 0.01, 0.02, 0.05, 0.10],
        "switch_lambda": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        "horizon_bars": [3, 5, 10, 20],
        "regime_cutoff": [1.0, 1.5, 2.0, 2.5, 999.0],
        "w_max": [0.25, 0.35, 0.50, 0.65, 0.80, 1.0],
        "top_n": [2, 3, 4, 5, 6],
        "min_threshold": [0.0, 0.5, 1.0, 2.0],
    }

    # Generate all combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    
    # Random sample if requested
    if args.samples > 0 and args.samples < len(combos):
        random.seed(42)
        combos = random.sample(combos, args.samples)
        print(f"Random sample: {len(combos)} / {len(list(itertools.product(*values)))} combinations")
    else:
        print(f"Full grid: {len(combos)} combinations")

    results: list[Result] = []
    for i, combo in enumerate(combos):
        overrides = {k: v for k, v in zip(keys, combo)}
        result = run_backtest(session, base_params, overrides, args.scale)
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(combos)} done...")

    # Sort by Sharpe
    by_sharpe = sorted(results, key=lambda r: r.sharpe, reverse=True)
    print(f"\n{'='*80}")
    print(f"TOP {args.top} BY SHARPE")
    print(f"{'='*80}")
    for i, r in enumerate(by_sharpe[: args.top]):
        print(f"{i+1}. Sharpe={r.sharpe:6.2f} | PnL=${r.net_pnl:12,.0f} | DD=${r.max_dd:12,.0f} | Costs=${r.costs:8,.0f}")
        print(f"   {r.params}")

    # Sort by PnL
    by_pnl = sorted(results, key=lambda r: r.net_pnl, reverse=True)
    print(f"\n{'='*80}")
    print(f"TOP {args.top} BY PnL")
    print(f"{'='*80}")
    for i, r in enumerate(by_pnl[: args.top]):
        print(f"{i+1}. PnL=${r.net_pnl:12,.0f} | Sharpe={r.sharpe:6.2f} | DD=${r.max_dd:12,.0f} | Costs=${r.costs:8,.0f}")
        print(f"   {r.params}")

    # Best overall (Sharpe > 2 with highest PnL)
    good = [r for r in results if r.sharpe > 2.0]
    if good:
        best = max(good, key=lambda r: r.net_pnl)
        print(f"\n{'='*80}")
        print("RECOMMENDED (Sharpe > 2, max PnL)")
        print(f"{'='*80}")
        print(f"Sharpe={best.sharpe:.2f} | PnL=${best.net_pnl:,.0f} | DD=${best.max_dd:,.0f}")
        print(f"Params: {best.params}")

        # Output as JSON snippet
        print("\nJSON snippet for strategy_params.json:")
        print(json.dumps(best.params, indent=2))


if __name__ == "__main__":
    main()

