from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from tradingai.data.connection import get_connection


def _sanitize_for_json(obj: object) -> object:
    """Replace float NaN/Infinity with None so the result is valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def save_regime_snapshot(
    payload: dict,
    *,
    as_of: datetime | None = None,
    database_url: str | None = None,
) -> int:
    when = as_of or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    clean = _sanitize_for_json(payload)
    with get_connection(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO regime_snapshots (as_of, payload)
            VALUES (%s, %s::jsonb)
            RETURNING id
            """,
            (when, json.dumps(clean, default=str)),
        ).fetchone()
    return int(row[0])


def get_latest_regime_snapshot(database_url: str | None = None) -> dict | None:
    with get_connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT id, as_of, created_at, payload
            FROM regime_snapshots
            ORDER BY as_of DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    payload = row[3]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "id": int(row[0]),
        "as_of": row[1],
        "created_at": row[2],
        "payload": payload,
    }
