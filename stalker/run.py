#!/usr/bin/env python3
"""
Stalker: Record all market data from RIT for offline replay.

Usage:
    python -m stalker.run
    python -m stalker.run --output ./my_data --poll 0.1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth import connect
from log_config import init_logging
from settings import settings
from stalker.recorder import Recorder


def main() -> None:
    parser = argparse.ArgumentParser(description='Record RIT market data')
    parser.add_argument('--output', '-o', default='data', help='Output directory')
    parser.add_argument('--poll', '-p', type=float, default=0.25, help='Poll interval in seconds')
    parser.add_argument('--api-key', '-k', help='Override API key')
    parser.add_argument('--api-host', '-H', help='Override API host')
    args = parser.parse_args()

    init_logging(console_level='INFO')

    api_key = args.api_key or settings.api_key
    api_host = args.api_host or settings.api_host

    client = connect(api_key, api_host)

    recorder = Recorder(
        client=client,
        output_dir=Path(args.output),
        poll_interval=args.poll,
    )
    recorder.run()


if __name__ == '__main__':
    main()
