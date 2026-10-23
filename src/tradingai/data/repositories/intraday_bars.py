from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from tradingai.data.connection import get_connection
from tradingai.data.repositories.daily_prices import is_finite_number

ET = ZoneInfo("America/New_York")
INTRADAY_BARS_TABLE = "intraday_bars"
INTRADAY_HISTORY_TABLE = "intraday_history_bars"
_ALLOWED_TABLES = frozenset({INTRADAY_BARS_TABLE, INTRADAY_HISTORY_TABLE})


def intraday_table(table: str) -> str:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unsupported intraday table: {table}")
    return table


def upsert_intraday_bars(
    ticker: str,
    rows: list[dict],
    *,
    table: str = INTRADAY_BARS_TABLE,
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
        name = intraday_table(table)
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {name} (ticker, bar_time, open, high, low, close, volume)
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


def _frame_from_rows(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        rows,
        columns=["bar_time", "open", "high", "low", "close", "volume"],
    )
    frame["bar_time"] = pd.to_datetime(frame["bar_time"], utc=True).dt.tz_convert(ET)
    frame = frame.set_index("bar_time").sort_index()
    return frame.astype(float)


def load_intraday_bars(
    ticker: str,
    *,
    table: str = INTRADAY_BARS_TABLE,
    start: datetime | None = None,
    end: datetime | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load 1m bars for one ticker, optionally bounded by ``[start, end)``."""
    name = intraday_table(table)
    clauses = ["ticker = %s"]
    params: list = [ticker.upper()]
    if start is not None:
        clauses.append("bar_time >= %s")
        params.append(start)
    if end is not None:
        clauses.append("bar_time < %s")
        params.append(end)
    sql = f"""
        SELECT bar_time, open, high, low, close, volume
        FROM {name}
        WHERE {" AND ".join(clauses)}
        ORDER BY bar_time ASC
    """
    with get_connection(database_url) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _frame_from_rows(rows)


def load_session_bars(
    ticker: str,
    *,
    session_date: datetime | date | None = None,
    table: str = INTRADAY_BARS_TABLE,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load 1m bars for one calendar date (America/New_York)."""
    if session_date is None:
        day = datetime.now(ET).date()
    elif isinstance(session_date, datetime):
        day = session_date.astimezone(ET).date()
    else:
        day = session_date

    start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=ET)
    end = start + timedelta(days=1)
    return load_intraday_bars(
        ticker,
        table=table,
        start=start,
        end=end,
        database_url=database_url,
    )


def session_has_bars(
    ticker: str,
    session_day: date,
    *,
    table: str = INTRADAY_BARS_TABLE,
    database_url: str | None = None,
) -> bool:
    """True when at least one bar is stored for this ticker and ET date."""
    name = intraday_table(table)
    start = datetime(session_day.year, session_day.month, session_day.day, tzinfo=ET)
    end = start + timedelta(days=1)
    with get_connection(database_url) as conn:
        row = conn.execute(
            f"""
            SELECT 1 FROM {name}
            WHERE ticker = %s AND bar_time >= %s AND bar_time < %s
            LIMIT 1
            """,
            (ticker.upper(), start, end),
        ).fetchone()
    return row is not None


def list_intraday_tickers(
    *,
    table: str = INTRADAY_BARS_TABLE,
    database_url: str | None = None,
) -> list[str]:
    name = intraday_table(table)
    with get_connection(database_url) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT ticker FROM {name} ORDER BY ticker"
        ).fetchall()
    return [str(row[0]) for row in rows]


def prune_intraday_bars(
    *,
    keep_days: int = 5,
    database_url: str | None = None,
) -> int:
    """Delete old rows from the live ``intraday_bars`` table only.

    ``intraday_history_bars`` is the training archive and is not pruned.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(keep_days)))
    with get_connection(database_url) as conn:
        cur = conn.execute(
            "DELETE FROM intraday_bars WHERE bar_time < %s",
            (cutoff,),
        )
        return int(cur.rowcount or 0)
