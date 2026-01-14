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


def run(params_path: str = None, scale: int = 1000) -> None:
    init_logging()

    # Load strategy params
    path = params_path or DEFAULT_PARAMS_PATH
    params = StrategyParams.load(path)
    logging.info('Loaded params from %s', path)
    logging.info('Position scale: %dx', scale)

    # Connect to RIT
    client = connect(settings.api_key, settings.api_host)
    wait_for_market(client)

    # Create runner with sizer
    sizer = FixedSizer(scale)
    runner = StrategyRunner(client, params, market, sizer=sizer)

    # Main loop
    while True:
        case = client.get_case()
        if case.get('status') != 'ACTIVE':
            logging.info('Market closed.')
            break

        portfolio = client.get_portfolio()
        runner.on_tick(portfolio, case)
        time.sleep(settings.poll_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run trading bot')
    parser.add_argument('--params', '-p', default=str(DEFAULT_PARAMS_PATH), help='Strategy params file')
    parser.add_argument('--scale', '-s', type=int, default=1000, help='Position size multiplier (default: 1000)')
    args = parser.parse_args()
    run(args.params, args.scale)
