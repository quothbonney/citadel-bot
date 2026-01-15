import argparse
import logging
import time

from auth import connect, wait_for_market
from log_config import init_logging
from market import market
from params import StrategyParams, DEFAULT_PARAMS_PATH
from runner import StrategyRunner
from settings import settings
from sizer import FixedSizer


def run(params_path: str = None, scale: int = 1000, verbose: bool = False,
        dashboard: bool = False) -> None:
    init_logging(console_level='DEBUG' if verbose else 'INFO')

    # Load strategy params
    path = params_path or DEFAULT_PARAMS_PATH
    params = StrategyParams.load(path)
    logging.info('Loaded params from %s', path)
    logging.info('Position scale: %dx', scale)

    # Start dashboard if enabled
    if dashboard:
        from dashboard import start_dashboard, update_state
        start_dashboard(port=5000)

    # Connect to RIT
    client = connect(settings.api_key, settings.api_host)
    wait_for_market(client)

    # Create runner with sizer
    sizer = FixedSizer(scale)
    runner = StrategyRunner(client, params, market, sizer=sizer)

    # Main loop
    tick_count = 0
    while True:
        case = client.get_case()
        if case.get('status') != 'ACTIVE':
            logging.info('Market closed.')
            break

        portfolio = client.get_portfolio()
        signals = runner.on_tick(portfolio, case)

        # Verbose: print signal info every 10 ticks
        if verbose and tick_count % 10 == 0:
            for sig in signals:
                logging.info('SIGNAL: %s | %s | %s', sig.strategy_id, sig.action, sig.reason)
            # Print current positions
            pos_str = ', '.join(f'{t}:{portfolio.get(t, {}).get("position", 0)}'
                               for t in ['ETF', 'AAA', 'BBB', 'CCC', 'DDD', 'IND'])
            logging.info('POSITIONS: %s', pos_str)

        # Update dashboard
        if dashboard:
            # Build positions dict for dashboard
            positions = {}
            for ticker in ['ETF', 'AAA', 'BBB', 'CCC', 'DDD', 'IND']:
                sec = portfolio.get(ticker, {})
                bid = sec.get('bid', 0)
                ask = sec.get('ask', 0)
                mid = (bid + ask) / 2 if bid and ask else sec.get('last', 0)
                positions[ticker] = {
                    'position': sec.get('position', 0),
                    'price': mid,
                }

            # Get PnL from portfolio (Rotman provides this)
            pnl = sum(sec.get('unrealized', 0) + sec.get('realized', 0)
                      for sec in portfolio.values() if isinstance(sec, dict))

            # Get active strategies from last signal
            active = []
            if signals and signals[0].reason:
                reason = signals[0].reason
                if 'active:' in reason:
                    active = [s.strip() for s in reason.split('active:')[1].split(',')]

            update_state(
                tick=case.get('tick', 0),
                period=case.get('period', 1),
                pnl=pnl,
                positions=positions,
                active=active,
            )

        tick_count += 1
        time.sleep(settings.poll_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run trading bot')
    parser.add_argument('--params', '-p', default=str(DEFAULT_PARAMS_PATH), help='Strategy params file')
    parser.add_argument('--scale', '-s', type=int, default=1000, help='Position size multiplier (default: 1000)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dashboard', '-d', action='store_true', help='Enable web dashboard at http://localhost:5000')
    args = parser.parse_args()
    run(args.params, args.scale, args.verbose, args.dashboard)
