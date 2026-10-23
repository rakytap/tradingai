from __future__ import annotations

import math
from datetime import date

from tradingai.data.connection import get_connection


def is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def upsert_daily_prices(
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
        if all(
            is_finite_number(row.get(key))
            for key in ("open", "high", "low", "close")
        )
    ]
    if not clean_rows:
        return 0
    with get_connection(database_url) as conn:
        params = [
            (
                ticker.upper(),
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            )
            for row in clean_rows
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO daily_prices (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                params,
            )
    return len(clean_rows)


def get_latest_date(ticker: str, database_url: str | None = None) -> date | None:
    with get_connection(database_url) as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker = %s",
            (ticker.upper(),),
        ).fetchone()
    return row[0] if row and row[0] else None


def get_forward_return(
    ticker: str,
    start_date: date,
    horizon_days: int,
    *,
    database_url: str | None = None,
) -> float | None:
    """Return percent change over the next `horizon_days` finite close sessions.

    Rows with non-finite closes are skipped so a single bad sync bar does not
    poison the outcome.
    """
    # Fetch a buffer so skipped NaN bars still leave enough finite sessions.
    fetch_limit = max(horizon_days * 4, horizon_days + 1) + 5
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT date, close
            FROM daily_prices
            WHERE ticker = %s AND date >= %s
            ORDER BY date ASC
            LIMIT %s
            """,
            (ticker.upper(), start_date, fetch_limit),
        ).fetchall()

    finite = [(d, float(c)) for d, c in rows if is_finite_number(c)]
    if len(finite) < horizon_days + 1:
        return None
    start_close = finite[0][1]
    end_close = finite[horizon_days][1]
    if start_close == 0:
        return None
    ret = (end_close / start_close - 1.0) * 100.0
    if not math.isfinite(ret):
        return None
    return ret


def list_closes(
    ticker: str,
    *,
    database_url: str | None = None,
) -> list[tuple[date, float]]:
    """Finite daily closes for one ticker, oldest first."""
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT date, close
            FROM daily_prices
            WHERE ticker = %s
            ORDER BY date ASC
            """,
            (ticker.upper(),),
        ).fetchall()
    return [(row[0], float(row[1])) for row in rows if is_finite_number(row[1])]


def delete_nonfinite_daily_prices(*, database_url: str | None = None) -> int:
    """Remove OHLC rows with non-finite prices (e.g. synced NaN closes)."""
    with get_connection(database_url) as conn:
        cur = conn.execute(
            """
            DELETE FROM daily_prices
            WHERE close::text IN ('NaN', 'Infinity', '-Infinity')
               OR open::text IN ('NaN', 'Infinity', '-Infinity')
               OR high::text IN ('NaN', 'Infinity', '-Infinity')
               OR low::text IN ('NaN', 'Infinity', '-Infinity')
            """
        )
        return cur.rowcount if cur.rowcount is not None else 0
