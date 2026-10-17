from __future__ import annotations

import argparse
import json

from tradingai.data.connection import initialize_schema
from tradingai.intraday.scan import scan_open_dip_bounces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan for morning open dips with bounce / reclaim signals."
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--dip-pct",
        type=float,
        help="Minimum plunge %% vs prior close (default TRADINGAI_OPEN_DIP_PCT / 2.0).",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        help="Minutes after RTH open to measure trough (default 60).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        help="Max watchlist size (default TRADINGAI_OPEN_DIP_WATCHLIST_CAP).",
    )
    parser.add_argument(
        "--buy-notional",
        type=float,
        help="USD stock notional for profit-target sell price (default 20000).",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        help="Net return on buy notional after fees (default (1/365)*(7/5)).",
    )
    parser.add_argument(
        "--sell-trail",
        type=float,
        help="Sell trailing-stop fraction (default = target_return / 2).",
    )
    parser.add_argument(
        "--profit-taker-buffer",
        type=float,
        help="Extra net return on profit-taker beyond target (default 0.0002 = 0.02%%).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Prefer live yfinance fetch over DB bars.",
    )
    args = parser.parse_args()
    initialize_schema()
    payload = scan_open_dip_bounces(
        tickers=args.tickers,
        dip_pct=args.dip_pct,
        window_minutes=args.window_minutes,
        prefer_db=not args.live,
        cap=args.cap,
        buy_notional_usd=args.buy_notional,
        target_return=args.target_return,
        sell_trail=args.sell_trail,
        profit_taker_buffer=args.profit_taker_buffer,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
