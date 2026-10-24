from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tradingai.data.connection import get_connection
from tradingai.data.repositories.daily_prices import is_finite_number
from tradingai.sim.config import get_schema_sql, get_sim_database_url

ET = ZoneInfo("America/New_York")


def initialize_sim_schema(database_url: str | None = None) -> None:
    sql = get_schema_sql()
    with get_connection(database_url or get_sim_database_url()) as conn:
        conn.execute(sql)


def upsert_sim_bars_1m(
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
    with get_connection(database_url or get_sim_database_url()) as conn:
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
                INSERT INTO sim_bars_1m (ticker, bar_time, open, high, low, close, volume)
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


def upsert_sim_daily_close(
    ticker: str,
    session_day: date,
    prior_close: float,
    *,
    database_url: str | None = None,
) -> None:
    if not is_finite_number(prior_close) or float(prior_close) <= 0:
        return
    with get_connection(database_url or get_sim_database_url()) as conn:
        conn.execute(
            """
            INSERT INTO sim_daily_closes (ticker, session_day, prior_close)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker, session_day) DO UPDATE SET
                prior_close = EXCLUDED.prior_close
            """,
            (ticker.upper(), session_day, float(prior_close)),
        )


def session_has_sim_bars(
    ticker: str,
    session_day: date,
    *,
    database_url: str | None = None,
) -> bool:
    start = datetime(session_day.year, session_day.month, session_day.day, 0, 0, tzinfo=ET)
    end = start.replace(hour=23, minute=59, second=59)
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM sim_bars_1m
            WHERE ticker = %s AND bar_time >= %s AND bar_time <= %s
            LIMIT 1
            """,
            (ticker.upper(), start, end),
        ).fetchone()
    return row is not None


def load_sim_bars_1m(
    ticker: str,
    *,
    session_day: date | None = None,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load 1m bars; when ``as_of`` is set, only bars with ``bar_time <= as_of``."""
    clauses = ["ticker = %s"]
    params: list[Any] = [ticker.upper()]
    if session_day is not None:
        day_start = datetime(
            session_day.year, session_day.month, session_day.day, 0, 0, tzinfo=ET
        )
        day_end = datetime(
            session_day.year, session_day.month, session_day.day, 23, 59, 59, tzinfo=ET
        )
        clauses.append("bar_time >= %s")
        params.append(day_start)
        clauses.append("bar_time <= %s")
        params.append(day_end)
    if start is not None:
        clauses.append("bar_time >= %s")
        params.append(start)
    if end is not None:
        clauses.append("bar_time < %s")
        params.append(end)
    if as_of is not None:
        clauses.append("bar_time <= %s")
        params.append(as_of)
    where = " AND ".join(clauses)
    with get_connection(database_url or get_sim_database_url()) as conn:
        rows = conn.execute(
            f"""
            SELECT bar_time, open, high, low, close, volume
            FROM sim_bars_1m
            WHERE {where}
            ORDER BY bar_time
            """,
            params,
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        rows,
        columns=["bar_time", "open", "high", "low", "close", "volume"],
    )
    frame["bar_time"] = pd.to_datetime(frame["bar_time"], utc=True).dt.tz_convert(ET)
    return frame.set_index("bar_time").sort_index().astype(float)


def load_prior_close(
    ticker: str,
    session_day: date,
    *,
    database_url: str | None = None,
) -> float | None:
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            SELECT prior_close
            FROM sim_daily_closes
            WHERE ticker = %s AND session_day = %s
            """,
            (ticker.upper(), session_day),
        ).fetchone()
    if row is None:
        return None
    value = float(row[0])
    return value if value > 0 else None


def list_sim_tickers(*, database_url: str | None = None) -> list[str]:
    with get_connection(database_url or get_sim_database_url()) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM sim_bars_1m ORDER BY ticker"
        ).fetchall()
    return [str(row[0]) for row in rows]


def list_sim_session_days(
    *,
    database_url: str | None = None,
) -> list[date]:
    with get_connection(database_url or get_sim_database_url()) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT (bar_time AT TIME ZONE 'America/New_York')::date AS session_day
            FROM sim_bars_1m
            ORDER BY session_day
            """
        ).fetchall()
    return [row[0] for row in rows]


def create_sim_run(
    *,
    params: dict,
    start_cash: float,
    database_url: str | None = None,
) -> int:
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            INSERT INTO sim_runs (params, start_cash)
            VALUES (%s::jsonb, %s)
            RETURNING run_id
            """,
            (json.dumps(params), float(start_cash)),
        ).fetchone()
    return int(row[0])


def finalize_sim_run(
    run_id: int,
    *,
    end_cash: float,
    realized_pnl: float,
    trade_count: int,
    win_count: int,
    summary: dict,
    database_url: str | None = None,
) -> None:
    with get_connection(database_url or get_sim_database_url()) as conn:
        conn.execute(
            """
            UPDATE sim_runs
            SET end_cash = %s,
                realized_pnl = %s,
                trade_count = %s,
                win_count = %s,
                summary = %s::jsonb
            WHERE run_id = %s
            """,
            (
                float(end_cash),
                float(realized_pnl),
                int(trade_count),
                int(win_count),
                json.dumps(summary),
                int(run_id),
            ),
        )


def insert_sim_position(
    run_id: int,
    *,
    ticker: str,
    qty: float,
    entry_price: float,
    entry_time: datetime,
    limit_price: float,
    stop_price: float,
    trail_activate_price: float | None = None,
    sell_trail: float | None = None,
    database_url: str | None = None,
) -> int:
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            INSERT INTO sim_positions (
                run_id, ticker, status, qty, entry_price, entry_time,
                limit_price, stop_price, trail_activate_price, sell_trail
            )
            VALUES (%s, %s, 'open', %s, %s, %s, %s, %s, %s, %s)
            RETURNING position_id
            """,
            (
                int(run_id),
                ticker.upper(),
                float(qty),
                float(entry_price),
                entry_time,
                float(limit_price),
                float(stop_price),
                float(trail_activate_price) if trail_activate_price is not None else None,
                float(sell_trail) if sell_trail is not None else None,
            ),
        ).fetchone()
    return int(row[0])


def close_sim_position(
    position_id: int,
    *,
    exit_price: float,
    exit_time: datetime,
    exit_reason: str,
    realized_pnl: float,
    database_url: str | None = None,
) -> None:
    with get_connection(database_url or get_sim_database_url()) as conn:
        conn.execute(
            """
            UPDATE sim_positions
            SET status = 'closed',
                exit_price = %s,
                exit_time = %s,
                exit_reason = %s,
                realized_pnl = %s
            WHERE position_id = %s
            """,
            (
                float(exit_price),
                exit_time,
                exit_reason,
                float(realized_pnl),
                int(position_id),
            ),
        )


def insert_sim_order(
    run_id: int,
    *,
    ticker: str,
    side: str,
    order_type: str,
    status: str,
    qty: float,
    created_at: datetime,
    limit_price: float | None = None,
    stop_price: float | None = None,
    filled_at: datetime | None = None,
    fill_price: float | None = None,
    position_id: int | None = None,
    database_url: str | None = None,
) -> int:
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            INSERT INTO sim_orders (
                run_id, ticker, side, order_type, status, qty,
                limit_price, stop_price, created_at, filled_at, fill_price, position_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING order_id
            """,
            (
                int(run_id),
                ticker.upper(),
                side,
                order_type,
                status,
                float(qty),
                float(limit_price) if limit_price is not None else None,
                float(stop_price) if stop_price is not None else None,
                created_at,
                filled_at,
                float(fill_price) if fill_price is not None else None,
                int(position_id) if position_id is not None else None,
            ),
        ).fetchone()
    return int(row[0])


def update_sim_order(
    order_id: int,
    *,
    status: str,
    filled_at: datetime | None = None,
    fill_price: float | None = None,
    position_id: int | None = None,
    database_url: str | None = None,
) -> None:
    with get_connection(database_url or get_sim_database_url()) as conn:
        conn.execute(
            """
            UPDATE sim_orders
            SET status = %s,
                filled_at = COALESCE(%s, filled_at),
                fill_price = COALESCE(%s, fill_price),
                position_id = COALESCE(%s, position_id)
            WHERE order_id = %s
            """,
            (
                status,
                filled_at,
                float(fill_price) if fill_price is not None else None,
                int(position_id) if position_id is not None else None,
                int(order_id),
            ),
        )


def insert_sim_fill(
    run_id: int,
    *,
    ticker: str,
    side: str,
    qty: float,
    price: float,
    filled_at: datetime,
    reason: str | None = None,
    order_id: int | None = None,
    database_url: str | None = None,
) -> int:
    with get_connection(database_url or get_sim_database_url()) as conn:
        row = conn.execute(
            """
            INSERT INTO sim_fills (
                run_id, order_id, ticker, side, qty, price, filled_at, reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING fill_id
            """,
            (
                int(run_id),
                int(order_id) if order_id is not None else None,
                ticker.upper(),
                side,
                float(qty),
                float(price),
                filled_at,
                reason,
            ),
        ).fetchone()
    return int(row[0])
