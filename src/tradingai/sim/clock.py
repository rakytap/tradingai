from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterator

import pandas as pd

from tradingai.intraday.config import ET
from tradingai.intraday.jobs.backfill_intraday_bars import (
    last_completed_session_day,
    weekday_sessions,
)
from tradingai.sim.repository import list_sim_session_days, load_sim_bars_1m


def as_of_bars(bars: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Return only bars with index <= ``as_of`` (no lookahead)."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ET)
    else:
        ts = ts.tz_convert(ET)
    return bars.loc[bars.index <= ts].copy()


def iter_session_bar_times(bars: pd.DataFrame) -> Iterator[pd.Timestamp]:
    """Yield each bar timestamp in chronological order."""
    if bars is None or bars.empty:
        return
    for ts in bars.index:
        yield pd.Timestamp(ts)


def session_days_for_download(days: int, *, end: date | None = None) -> list[date]:
    end_day = end or last_completed_session_day()
    return weekday_sessions(days, end=end_day)


def session_days_in_sim_db(*, database_url: str | None = None) -> list[date]:
    return list_sim_session_days(database_url=database_url)


def load_session_timeline(
    tickers: list[str],
    session_day: date,
    *,
    database_url: str | None = None,
) -> list[pd.Timestamp]:
    """Union of all 1m bar times for ``tickers`` on ``session_day``, sorted."""
    times: set[pd.Timestamp] = set()
    for ticker in tickers:
        bars = load_sim_bars_1m(
            ticker,
            session_day=session_day,
            database_url=database_url,
        )
        for ts in bars.index:
            times.add(pd.Timestamp(ts))
    return sorted(times)


def entry_window_start(session_day: date, *, window_minutes: int) -> datetime:
    """First decision time: RTH open + dip window."""
    open_ts = datetime(
        session_day.year,
        session_day.month,
        session_day.day,
        9,
        30,
        tzinfo=ET,
    )
    return open_ts + timedelta(minutes=max(5, int(window_minutes)))


def entry_cutoff_time(
    session_day: date,
    *,
    hour: int,
    minute: int,
) -> datetime:
    return datetime(
        session_day.year,
        session_day.month,
        session_day.day,
        int(hour),
        int(minute),
        tzinfo=ET,
    )
