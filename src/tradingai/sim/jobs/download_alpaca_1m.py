"""Download ~60 weekdays of Alpaca 1m RTH bars into ``tradingai_sim``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from tradingai.data.config import get_sync_tickers
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.intraday.bars import bars_to_rows, filter_rth_session
from tradingai.intraday.config import ET
from tradingai.providers.alpaca_provider import fetch_stock_bars
from tradingai.sim.clock import session_days_for_download
from tradingai.sim.config import (
    DEFAULT_SIM_CONFIG_PATH,
    clear_sim_config_cache,
    get_download_days,
    get_download_pace_seconds,
    get_sim_database_url,
)
from tradingai.sim.repository import (
    initialize_sim_schema,
    session_has_sim_bars,
    upsert_sim_bars_1m,
    upsert_sim_daily_close,
)


def _prior_close_from_daily(ticker: str, session_day: date) -> float | None:
    """Fetch prior session close via Alpaca daily bars."""
    start = datetime(
        session_day.year, session_day.month, session_day.day, tzinfo=ET
    ) - timedelta(days=10)
    end = datetime(
        session_day.year, session_day.month, session_day.day, 9, 30, tzinfo=ET
    )
    try:
        daily = fetch_stock_bars(
            ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
    except Exception:
        return None
    if daily.empty:
        return None
    # Bars strictly before session_day.
    prior = daily.loc[daily.index.date < session_day]
    if prior.empty:
        return None
    return float(prior["close"].iloc[-1])


def _prior_close_from_prev_session(
    ticker: str,
    session_day: date,
    sessions: list[date],
    *,
    database_url: str,
) -> float | None:
    """Fallback: last 1m close of the previous weekday already in DB."""
    try:
        idx = sessions.index(session_day)
    except ValueError:
        return None
    if idx <= 0:
        return None
    prev = sessions[idx - 1]
    from tradingai.sim.repository import load_sim_bars_1m

    bars = load_sim_bars_1m(ticker, session_day=prev, database_url=database_url)
    if bars.empty:
        return None
    return float(bars["close"].iloc[-1])


def download_alpaca_1m(
    tickers: list[str] | None = None,
    *,
    days: int | None = None,
    pace_seconds: float | None = None,
    database_url: str | None = None,
    config_path: str | None = None,
) -> dict:
    db_url = database_url or get_sim_database_url(config_path=config_path)
    day_count = int(days) if days is not None else get_download_days(config_path=config_path)
    pace = (
        float(pace_seconds)
        if pace_seconds is not None
        else get_download_pace_seconds(config_path=config_path)
    )
    initialize_sim_schema(db_url)
    if tickers:
        universe = list(dict.fromkeys(t.upper() for t in tickers if t.strip()))
    else:
        universe = list(get_sync_tickers())
    sessions = session_days_for_download(day_count)
    total = len(universe) * len(sessions)
    log_start("tradingai-sim-download", len(universe))

    written_bars = 0
    stored_days = 0
    skipped = 0
    empty = 0
    closes_written = 0
    errors: list[str] = []
    index = 0
    timeframe = TimeFrame(1, TimeFrameUnit.Minute)

    for ticker in universe:
        for day in sessions:
            index += 1
            if session_has_sim_bars(ticker, day, database_url=db_url):
                skipped += 1
                log_progress(index, total, ticker, f"{day.isoformat()} skipped")
                continue
            start = datetime(day.year, day.month, day.day, 9, 0, tzinfo=ET)
            end = datetime(day.year, day.month, day.day, 16, 30, tzinfo=ET)
            try:
                frame = fetch_stock_bars(
                    ticker,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                )
            except Exception as exc:
                message = f"{ticker} {day.isoformat()}: {exc}"
                errors.append(message)
                log_error(index, total, ticker, f"{day.isoformat()} {exc}")
                if pace > 0:
                    time.sleep(pace)
                continue

            session = filter_rth_session(frame, session_day=day)
            if session.empty:
                empty += 1
                log_progress(index, total, ticker, f"{day.isoformat()} empty")
            else:
                rows = bars_to_rows(session)
                written = upsert_sim_bars_1m(ticker, rows, database_url=db_url)
                written_bars += written
                stored_days += 1
                prior = _prior_close_from_daily(ticker, day)
                if prior is None:
                    prior = _prior_close_from_prev_session(
                        ticker, day, sessions, database_url=db_url
                    )
                if prior is not None:
                    upsert_sim_daily_close(
                        ticker, day, prior, database_url=db_url
                    )
                    closes_written += 1
                log_progress(
                    index,
                    total,
                    ticker,
                    f"{day.isoformat()} {written} bars",
                )

            if pace > 0:
                time.sleep(pace)

    log_done(
        "tradingai-sim-download",
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
        "closes_written": closes_written,
        "skipped": skipped,
        "empty": empty,
        "errors": errors[:50],
        "database_url": db_url,
        "table": "sim_bars_1m",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Alpaca 1m RTH bars into tradingai_sim. "
            f"Defaults from {DEFAULT_SIM_CONFIG_PATH.name} (not .env)."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to sim TOML (default: {DEFAULT_SIM_CONFIG_PATH}).",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Weekday sessions to download (default from config).",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=None,
        help="Seconds between Alpaca requests (default from config).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override database.url from sim TOML.",
    )
    args = parser.parse_args()
    clear_sim_config_cache()
    try:
        result = download_alpaca_1m(
            args.tickers,
            days=args.days,
            pace_seconds=args.pace,
            database_url=args.database_url,
            config_path=args.config,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
