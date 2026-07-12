from __future__ import annotations

from datetime import datetime, timezone

from tradingai.data.connection import get_connection


def upsert_stock(
    ticker: str,
    *,
    sector: str | None = None,
    industry: str | None = None,
    market_cap: int | None = None,
    database_url: str | None = None,
) -> None:
    with get_connection(database_url) as conn:
        conn.execute(
            """
            INSERT INTO stocks (ticker, sector, industry, market_cap, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                market_cap = EXCLUDED.market_cap,
                updated_at = EXCLUDED.updated_at
            """,
            (
                ticker.upper(),
                sector,
                industry,
                market_cap,
                datetime.now(timezone.utc),
            ),
        )


def get_stock(ticker: str, database_url: str | None = None) -> dict | None:
    with get_connection(database_url) as conn:
        row = conn.execute(
            "SELECT ticker, sector, industry, market_cap, updated_at FROM stocks WHERE ticker = %s",
            (ticker.upper(),),
        ).fetchone()
    if row is None:
        return None
    return {
        "ticker": row[0],
        "sector": row[1],
        "industry": row[2],
        "market_cap": row[3],
        "updated_at": row[4],
    }


def list_tickers(database_url: str | None = None) -> list[str]:
    with get_connection(database_url) as conn:
        rows = conn.execute("SELECT ticker FROM stocks ORDER BY ticker").fetchall()
    return [row[0] for row in rows]
