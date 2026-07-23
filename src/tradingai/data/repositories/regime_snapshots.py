from __future__ import annotations

import json
from datetime import datetime, timezone

from tradingai.data.connection import get_connection


def save_regime_snapshot(
    payload: dict,
    *,
    as_of: datetime | None = None,
    database_url: str | None = None,
) -> int:
    when = as_of or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    with get_connection(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO regime_snapshots (as_of, payload)
            VALUES (%s, %s::jsonb)
            RETURNING id
            """,
            (when, json.dumps(payload, default=str)),
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
