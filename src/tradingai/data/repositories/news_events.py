from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from tradingai.data.connection import get_connection


SENTIMENT_SNAPSHOT_HEADLINE = "__sentiment_snapshot__"
SENTIMENT_SNAPSHOT_SOURCE = "tradingai-predict"


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


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_db_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, str):
        try:
            # Handles "...Z" only after conversion.
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return _to_utc(parsed)
        except ValueError:
            return None
    return None


def get_latest_sentiment_snapshot(
    ticker: str,
    *,
    max_age_minutes: int = 120,
    database_url: str | None = None,
) -> dict | None:
    """Return latest cached analyze_news_sentiment payload, if fresh enough."""
    with get_connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT timestamp, raw_json
            FROM news_events
            WHERE ticker = %s
              AND source = %s
              AND headline = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (ticker.upper(), SENTIMENT_SNAPSHOT_SOURCE, SENTIMENT_SNAPSHOT_HEADLINE),
        ).fetchone()
    if row is None:
        return None

    ts = _parse_db_timestamp(row[0])
    if ts is None:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    if ts < cutoff:
        return None

    payload = row[1] or {}
    # Some drivers return JSONB as str; keep it robust.
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def save_sentiment_snapshot(
    ticker: str,
    *,
    payload: dict,
    database_url: str | None = None,
) -> None:
    """Persist a full analyze_news_sentiment payload as a single cached snapshot row."""
    as_of = payload.get("as_of")
    ts = _parse_db_timestamp(as_of) or datetime.now(timezone.utc)
    sentiment = payload.get("sentiment")
    try:
        sentiment_f = float(sentiment) if sentiment is not None else None
    except (TypeError, ValueError):
        sentiment_f = None

    upsert_news_event(
        ticker,
        timestamp=ts,
        headline=SENTIMENT_SNAPSHOT_HEADLINE,
        sentiment=sentiment_f,
        impact="sentiment_snapshot",
        source=SENTIMENT_SNAPSHOT_SOURCE,
        raw_json=payload,
        database_url=database_url,
    )
