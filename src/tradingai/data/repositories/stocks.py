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
    """Insert or update a stock row.

    ``None`` metadata fields do not wipe existing values (so jobs like
    ``tradingai-sync-news`` that only ensure the ticker exists cannot clear
    sector/industry populated by ``tradingai-sync-stocks``).
    """
    with get_connection(database_url) as conn:
        conn.execute(
            """
            INSERT INTO stocks (ticker, sector, industry, market_cap, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                sector = COALESCE(EXCLUDED.sector, stocks.sector),
                industry = COALESCE(EXCLUDED.industry, stocks.industry),
                market_cap = COALESCE(EXCLUDED.market_cap, stocks.market_cap),
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


def list_sectors(
    *,
    exclude_etf: bool = True,
    database_url: str | None = None,
) -> list[str]:
    with get_connection(database_url) as conn:
        if exclude_etf:
            rows = conn.execute(
                """
                SELECT DISTINCT sector
                FROM stocks
                WHERE sector IS NOT NULL
                  AND TRIM(sector) <> ''
                  AND LOWER(TRIM(sector)) <> 'etf'
                ORDER BY sector
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT sector
                FROM stocks
                WHERE sector IS NOT NULL AND TRIM(sector) <> ''
                ORDER BY sector
                """
            ).fetchall()
    return [str(row[0]) for row in rows]


def list_tickers_by_sector(
    sector: str,
    *,
    exclude_etf: bool = True,
    database_url: str | None = None,
) -> list[str]:
    with get_connection(database_url) as conn:
        if exclude_etf:
            rows = conn.execute(
                """
                SELECT ticker
                FROM stocks
                WHERE sector = %s
                  AND (industry IS NULL OR LOWER(TRIM(industry)) <> 'exchange traded fund')
                ORDER BY ticker
                """,
                (sector,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ticker FROM stocks WHERE sector = %s ORDER BY ticker
                """,
                (sector,),
            ).fetchall()
    return [str(row[0]) for row in rows]
