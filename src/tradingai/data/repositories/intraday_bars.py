from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from tradingai.data.connection import get_connection
from tradingai.data.repositories.daily_prices import is_finite_number


def upsert_intraday_bars(
    ticker: str,
    rows: list[dict],
    *,
    database_url: str | None = None,
) -> int:
    if not rows:
        return 0
    clean_rows = [
        row
        for row in rows
        if row.get("bar_time") is not None
        and all(is_finite_number(row.get(key)) for key in ("open", "high", "low", "close"))
    ]
    if not clean_rows:
        return 0
    with get_connection(database_url) as conn:
        params = [
            (
                ticker.upper(),
                row["bar_time"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]) if is_finite_number(row.get("volume")) else 0.0,
            )
            for row in clean_rows
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO intraday_bars (ticker, bar_time, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, bar_time) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                params,
            )
    return len(clean_rows)


def load_session_bars(
    ticker: str,
    *,
    session_date: datetime | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load 1m bars for one RTH calendar date (America/New_York)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    if session_date is None:
        day = datetime.now(et).date()
    elif isinstance(session_date, datetime):
        day = session_date.astimezone(et).date()
    else:
        day = session_date  # type: ignore[assignment]

    start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=et)
    end = start + timedelta(days=1)
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT bar_time, open, high, low, close, volume
            FROM intraday_bars
            WHERE ticker = %s AND bar_time >= %s AND bar_time < %s
            ORDER BY bar_time ASC
            """,
            (ticker.upper(), start, end),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        rows,
        columns=["bar_time", "open", "high", "low", "close", "volume"],
    )
    frame["bar_time"] = pd.to_datetime(frame["bar_time"], utc=True).dt.tz_convert(et)
    frame = frame.set_index("bar_time").sort_index()
    return frame.astype(float)


def prune_intraday_bars(
    *,
    keep_days: int = 5,
    database_url: str | None = None,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(keep_days)))
    with get_connection(database_url) as conn:
        cur = conn.execute(
            "DELETE FROM intraday_bars WHERE bar_time < %s",
            (cutoff,),
        )
        return int(cur.rowcount or 0)
