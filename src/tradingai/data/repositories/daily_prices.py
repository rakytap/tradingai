from __future__ import annotations

from datetime import date

from tradingai.data.connection import get_connection


def upsert_daily_prices(
    ticker: str,
    rows: list[dict],
    *,
    database_url: str | None = None,
) -> int:
    if not rows:
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
            for row in rows
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
    return len(rows)


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
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT date, close
            FROM daily_prices
            WHERE ticker = %s AND date >= %s
            ORDER BY date ASC
            LIMIT %s
            """,
            (ticker.upper(), start_date, horizon_days + 1),
        ).fetchall()
    if len(rows) < horizon_days + 1:
        return None
    start_close = float(rows[0][1])
    end_close = float(rows[horizon_days][1])
    if start_close == 0:
        return None
    return (end_close / start_close - 1.0) * 100.0
