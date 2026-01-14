import logging
import sys
import time

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi


def connect(api_key: str, api_host: str) -> RotmanInteractiveTraderApi:
    client = RotmanInteractiveTraderApi(api_key=api_key, api_host=api_host)
    trader = client.get_trader()
    if isinstance(trader, dict) and trader.get('code'):
        logging.error('Failed to connect: %s', trader)
        sys.exit(1)
    logging.info('Connected as %s %s', trader.get('first_name'), trader.get('last_name'))
    return client


def wait_for_market(client: RotmanInteractiveTraderApi) -> dict:
    while True:
        case = client.get_case()
        if case.get('status') == 'ACTIVE':
            return case
        logging.info('Waiting for market to open...')
        time.sleep(1)
