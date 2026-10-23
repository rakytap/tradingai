"""Backfill 1-minute regular-session bars into ``intraday_history_bars``.

IBKR serves at most one day of 1-minute bars per request. This job walks
completed weekday sessions backward, skips ticker-days already stored, and
does not prune the history table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.intraday_bars import (
    INTRADAY_HISTORY_TABLE,
    session_has_bars,
    upsert_intraday_bars,
)
from tradingai.intraday.bars import bars_to_rows, filter_rth_session
from tradingai.intraday.config import ET, get_ibkr_host, get_ibkr_port
from tradingai.providers.ibkr_tws import (
    connect_ib,
    fetch_intraday_ohlcv_ibkr,
    qualify_equity,
)

DEFAULT_BACKFILL_DAYS = 60
DEFAULT_BACKFILL_PACE_SECONDS = 10.0
DEFAULT_BACKFILL_RETRIES = 3


def last_completed_session_day(now: datetime | None = None) -> date:
    """Most recent weekday whose 16:00 ET close is already in the past."""
    current = now or datetime.now(ET)
    if current.tzinfo is None:
        current = current.replace(tzinfo=ET)
    else:
        current = current.astimezone(ET)
    day = current.date()
    close = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    if current >= close and day.weekday() < 5:
        return day
    cursor = day - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def weekday_sessions(n: int, *, end: date) -> list[date]:
    """``n`` weekdays ending at ``end`` (or the prior weekday), oldest first."""
    count = max(1, int(n))
    cursor = end
    found: list[date] = []
    while len(found) < count:
        if cursor.weekday() < 5:
            found.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(found))


def classify_fetch_error(exc: BaseException) -> str:
    """``pacing`` retries, ``empty`` is a session with no bars, else ``error``."""
    text = str(exc).lower()
    if "pacing" in text:
        return "pacing"
    if "no ohlcv" in text or "no data" in text:
        return "empty"
    return "error"


def _fetch_session_day(ib, ticker: str, day: date, contract, *, max_retries: int, pace_seconds: float):
    end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    attempts = max(1, int(max_retries))
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fetch_intraday_ohlcv_ibkr(
                ib,
                ticker,
                period="1d",
                interval="1m",
                end_datetime=end,
                contract=contract,
            )
        except Exception as exc:
            kind = classify_fetch_error(exc)
            if kind == "pacing" and attempt + 1 < attempts:
                last_error = exc
                time.sleep(max(0.0, pace_seconds) * (attempt + 1))
                continue
            if kind == "empty":
                return None
            raise
    if last_error is not None:
        raise last_error
    return None


def backfill_intraday_bars(
    tickers: list[str] | None = None,
    *,
    days: int = DEFAULT_BACKFILL_DAYS,
    pace_seconds: float = DEFAULT_BACKFILL_PACE_SECONDS,
    max_retries: int = DEFAULT_BACKFILL_RETRIES,
) -> dict:
    initialize_schema()
    if tickers:
        universe = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
    else:
        universe = list(get_sync_tickers())
    sessions = weekday_sessions(days, end=last_completed_session_day())
    total = len(universe) * len(sessions)
    log_start("tradingai-backfill-intraday", len(universe))
    written_bars = 0
    stored_days = 0
    skipped = 0
    empty = 0
    errors: list[str] = []
    index = 0

    try:
        connection = connect_ib()
        ib = connection.__enter__()
    except Exception as exc:
        message = (
            f"IBKR TWS connection failed ({get_ibkr_host()}:{get_ibkr_port()}): {exc}. "
            "Ensure TWS/Gateway is running with API enabled."
        )
        raise RuntimeError(message) from exc

    try:
        ib.RaiseRequestErrors = True
        for ticker in universe:
            try:
                contract = qualify_equity(ib, ticker)
            except Exception as exc:
                message = f"{ticker}: {exc}"
                errors.append(message)
                log_error(index + 1, total, ticker, str(exc))
                index += len(sessions)
                continue
            for day in sessions:
                index += 1
                if session_has_bars(
                    ticker,
                    day,
                    table=INTRADAY_HISTORY_TABLE,
                ):
                    skipped += 1
                    log_progress(index, total, ticker, f"{day.isoformat()} skipped")
                    continue
                try:
                    frame = _fetch_session_day(
                        ib,
                        ticker,
                        day,
                        contract,
                        max_retries=max_retries,
                        pace_seconds=pace_seconds,
                    )
                except Exception as exc:
                    errors.append(f"{ticker} {day.isoformat()}: {exc}")
                    log_error(index, total, ticker, f"{day.isoformat()} {exc}")
                    frame = None
                else:
                    if frame is None:
                        empty += 1
                        log_progress(index, total, ticker, f"{day.isoformat()} empty")
                    else:
                        session = filter_rth_session(frame, session_day=day)
                        if session.empty:
                            empty += 1
                            log_progress(index, total, ticker, f"{day.isoformat()} empty")
                        else:
                            rows = bars_to_rows(session)
                            written = upsert_intraday_bars(
                                ticker,
                                rows,
                                table=INTRADAY_HISTORY_TABLE,
                            )
                            written_bars += written
                            stored_days += 1
                            log_progress(
                                index,
                                total,
                                ticker,
                                f"{day.isoformat()} {written} bars",
                            )
                if pace_seconds > 0:
                    time.sleep(pace_seconds)
    finally:
        connection.__exit__(None, None, None)

    log_done(
        "tradingai-backfill-intraday",
        stored_days,
        len(universe),
        skipped=skipped,
        empty=empty,
        failed=len(errors),
    )
    return {
        "tickers": universe,
        "sessions": [day.isoformat() for day in sessions],
        "stored_days": stored_days,
        "bars_written": written_bars,
        "skipped": skipped,
        "empty": empty,
        "errors": errors,
        "table": INTRADAY_HISTORY_TABLE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill IBKR 1m regular-session bars into intraday_history_bars."
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_BACKFILL_DAYS,
        help=f"Weekday sessions to request, walking backward (default {DEFAULT_BACKFILL_DAYS}).",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_BACKFILL_PACE_SECONDS,
        help=f"Seconds to wait after each IBKR request (default {DEFAULT_BACKFILL_PACE_SECONDS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_BACKFILL_RETRIES,
        help=f"Attempts per session when IBKR reports pacing (default {DEFAULT_BACKFILL_RETRIES}).",
    )
    args = parser.parse_args()
    try:
        result = backfill_intraday_bars(
            args.tickers,
            days=args.days,
            pace_seconds=args.pace,
            max_retries=args.retries,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
