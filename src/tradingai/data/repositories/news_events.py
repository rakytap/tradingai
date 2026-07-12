from __future__ import annotations

import json
from datetime import datetime

from tradingai.data.connection import get_connection


def upsert_news_event(
    ticker: str,
    *,
    timestamp: datetime,
    headline: str,
    sentiment: float | None = None,
    impact: str | None = None,
    source: str | None = None,
    raw_json: dict | None = None,
    database_url: str | None = None,
) -> None:
    with get_connection(database_url) as conn:
        conn.execute(
            """
            INSERT INTO news_events (
                ticker, timestamp, headline, sentiment, impact, source, raw_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (ticker, timestamp, headline) DO UPDATE SET
                sentiment = EXCLUDED.sentiment,
                impact = EXCLUDED.impact,
                source = EXCLUDED.source,
                raw_json = EXCLUDED.raw_json
            """,
            (
                ticker.upper(),
                timestamp,
                headline,
                sentiment,
                impact,
                source,
                json.dumps(raw_json or {}),
            ),
        )
