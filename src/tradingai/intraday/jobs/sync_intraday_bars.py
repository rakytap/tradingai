from __future__ import annotations

import argparse
import json
import sys
import time

from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.intraday_bars import prune_intraday_bars, upsert_intraday_bars
from tradingai.intraday.bars import bars_to_rows
from tradingai.intraday.config import (
    DEFAULT_KEEP_DAYS,
    get_bar_interval,
    get_bar_period,
    get_ibkr_host,
    get_ibkr_pace_seconds,
    get_ibkr_port,
)
from tradingai.intraday.watchlist import bullish_sector_watchlist
from tradingai.providers.ibkr_tws import connect_ib, fetch_intraday_ohlcv_ibkr


def sync_intraday_bars(
    tickers: list[str] | None = None,
    *,
    cap: int | None = None,
    keep_days: int = DEFAULT_KEEP_DAYS,
) -> dict:
    initialize_schema()
    universe = bullish_sector_watchlist(tickers=tickers, cap=cap)
    total = len(universe)
    log_start("tradingai-sync-intraday", total)
    synced = 0
    bars_written = 0
    errors: list[str] = []
    pace = get_ibkr_pace_seconds()
    interval = get_bar_interval()
    period = get_bar_period()

    try:
        with connect_ib() as ib:
            for index, ticker in enumerate(universe, start=1):
                try:
                    frame = fetch_intraday_ohlcv_ibkr(
                        ib,
                        ticker,
                        period=period,
                        interval=interval,
                    )
                    rows = bars_to_rows(frame)
                    written = upsert_intraday_bars(ticker, rows)
                    bars_written += written
                    synced += 1
                    log_progress(
                        index,
                        total,
                        ticker,
                        f"{written} bars (ibkr {interval})",
                    )
                except Exception as exc:
                    errors.append(f"{ticker}: {exc}")
                    log_error(index, total, ticker, str(exc))
                if pace > 0 and index < total:
                    time.sleep(pace)
    except Exception as exc:
        message = (
            f"IBKR TWS connection failed ({get_ibkr_host()}:{get_ibkr_port()}): {exc}. "
            "Ensure TWS/Gateway is running with API enabled."
        )
        raise RuntimeError(message) from exc

    pruned = prune_intraday_bars(keep_days=keep_days)
    log_done("tradingai-sync-intraday", synced, total, failed=len(errors))
    return {
        "synced": synced,
        "bars_written": bars_written,
        "pruned_rows": pruned,
        "universe": universe,
        "interval": interval,
        "period": period,
        "source": "ibkr_tws",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync IBKR TWS 1m intraday bars for the open-dip watchlist."
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--cap",
        type=int,
        help="Max tickers (default TRADINGAI_OPEN_DIP_WATCHLIST_CAP / 1000).",
    )
    parser.add_argument(
        "--keep-days",
        type=int,
        default=DEFAULT_KEEP_DAYS,
        help=f"Prune bars older than this many days (default {DEFAULT_KEEP_DAYS}).",
    )
    args = parser.parse_args()
    try:
        result = sync_intraday_bars(
            args.tickers,
            cap=args.cap,
            keep_days=args.keep_days,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
