from __future__ import annotations

import json
from datetime import datetime, timezone

from tradingai.data.connection import get_connection
from tradingai.data.repositories.stocks import upsert_stock


def save_news_graph_events(
    ticker: str,
    *,
    events: list[dict],
    observed_at: datetime | None = None,
    database_url: str | None = None,
) -> int:
    """Replace recent graph events for a ticker with the latest extraction batch."""
    if not events:
        return 0
    when = observed_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    symbol = ticker.upper()
    upsert_stock(symbol, database_url=database_url)
    written = 0
    with get_connection(database_url) as conn:
        # Keep history bounded: delete same-day rows for this ticker before insert.
        conn.execute(
            """
            DELETE FROM news_graph_events
            WHERE ticker = %s
              AND observed_at::date = %s::date
            """,
            (symbol, when),
        )
        for event in events:
            conn.execute(
                """
                INSERT INTO news_graph_events (
                    ticker, observed_at, event_type, polarity, themes, sectors,
                    evidence, confidence, raw_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    symbol,
                    when,
                    str(event.get("event_type", "unknown")),
                    int(event.get("polarity", 0)),
                    list(event.get("themes") or []),
                    list(event.get("sectors") or []),
                    event.get("evidence"),
                    event.get("confidence"),
                    json.dumps(event, default=str),
                ),
            )
            written += 1
    return written


def get_news_graph_events(
    ticker: str,
    *,
    limit: int = 20,
    database_url: str | None = None,
) -> list[dict]:
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT id, ticker, observed_at, event_type, polarity, themes, sectors,
                   evidence, confidence, raw_json
            FROM news_graph_events
            WHERE ticker = %s
            ORDER BY observed_at DESC, id DESC
            LIMIT %s
            """,
            (ticker.upper(), limit),
        ).fetchall()

    results: list[dict] = []
    for row in rows:
        raw = row[9] or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        results.append(
            {
                "id": int(row[0]),
                "ticker": row[1],
                "observed_at": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
                "event_type": row[3],
                "polarity": int(row[4]),
                "themes": list(row[5] or []),
                "sectors": list(row[6] or []),
                "evidence": row[7],
                "confidence": float(row[8]) if row[8] is not None else None,
                "raw": raw,
            }
        )
    return results
