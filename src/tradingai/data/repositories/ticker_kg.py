from __future__ import annotations

import json
from datetime import datetime, timezone

from tradingai.data.connection import get_connection
from tradingai.data.repositories.stocks import upsert_stock
from tradingai.news.entity_graph import ENTITY_GRAPH, TickerNode, get_node, node_from_dict


RELATION_FIELDS = ("peers", "suppliers", "customers", "thematic_proxies", "commodities")


def _row_from_node(node: TickerNode) -> dict:
    return {
        "ticker": node.ticker.upper(),
        "sector": node.sector,
        "peers": list(node.peers),
        "suppliers": list(node.suppliers),
        "customers": list(node.customers),
        "thematic_proxies": list(node.thematic_proxies),
        "commodities": list(node.commodities),
        "rate_sensitive": node.rate_sensitive,
        "fx_sensitive": node.fx_sensitive,
        "duration_bucket": node.duration_bucket,
        "notes": node.notes,
        "raw_json": node.to_dict(),
    }


def _dict_from_row(row) -> dict:
    raw = row[11] or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return {
        "ticker": row[0],
        "sector": row[1],
        "peers": list(row[2] or []),
        "suppliers": list(row[3] or []),
        "customers": list(row[4] or []),
        "thematic_proxies": list(row[5] or []),
        "commodities": list(row[6] or []),
        "rate_sensitive": bool(row[7]),
        "fx_sensitive": bool(row[8]),
        "duration_bucket": row[9],
        "notes": row[10],
        "raw": raw,
        "updated_at": row[12].isoformat() if hasattr(row[12], "isoformat") else row[12],
    }


def fetch_ticker_kg_row(
    ticker: str,
    *,
    database_url: str | None = None,
) -> dict | None:
    """Return DB row only (no static fallback). None if missing or DB unavailable."""
    try:
        with get_connection(database_url) as conn:
            row = conn.execute(
                """
                SELECT ticker, sector, peers, suppliers, customers, thematic_proxies,
                       commodities, rate_sensitive, fx_sensitive, duration_bucket,
                       notes, raw_json, updated_at
                FROM ticker_kg_nodes
                WHERE ticker = %s
                """,
                (ticker.upper(),),
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return _dict_from_row(row)


def _merge_preserve_overlays(seed: TickerNode, existing: dict) -> TickerNode:
    """Keep non-empty DB relation arrays; fill other fields from seed."""
    payload = seed.to_dict()
    for field in RELATION_FIELDS:
        current = list(existing.get(field) or [])
        if current:
            payload[field] = current
    raw = existing.get("raw") or {}
    if isinstance(raw, dict) and raw.get("source") == "llm_curator" and existing.get("notes"):
        payload["notes"] = existing["notes"]
    return node_from_dict(payload)


def upsert_ticker_kg_node(
    node: TickerNode,
    *,
    database_url: str | None = None,
    preserve_overlays: bool = False,
    raw_json: dict | None = None,
) -> None:
    existing = None
    to_write = node
    if preserve_overlays:
        existing = fetch_ticker_kg_row(node.ticker, database_url=database_url)
        if existing is not None:
            to_write = _merge_preserve_overlays(node, existing)

    payload = _row_from_node(to_write)
    if raw_json is not None:
        payload["raw_json"] = raw_json
    elif (
        preserve_overlays
        and existing
        and isinstance(existing.get("raw"), dict)
        and existing["raw"].get("source") == "llm_curator"
    ):
        payload["raw_json"] = {
            **existing["raw"],
            **payload["raw_json"],
            "source": "llm_curator",
            "seed_merged": True,
        }

    upsert_stock(payload["ticker"], database_url=database_url)
    with get_connection(database_url) as conn:
        conn.execute(
            """
            INSERT INTO ticker_kg_nodes (
                ticker, sector, peers, suppliers, customers, thematic_proxies,
                commodities, rate_sensitive, fx_sensitive, duration_bucket,
                notes, raw_json, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            ON CONFLICT (ticker) DO UPDATE SET
                sector = EXCLUDED.sector,
                peers = EXCLUDED.peers,
                suppliers = EXCLUDED.suppliers,
                customers = EXCLUDED.customers,
                thematic_proxies = EXCLUDED.thematic_proxies,
                commodities = EXCLUDED.commodities,
                rate_sensitive = EXCLUDED.rate_sensitive,
                fx_sensitive = EXCLUDED.fx_sensitive,
                duration_bucket = EXCLUDED.duration_bucket,
                notes = EXCLUDED.notes,
                raw_json = EXCLUDED.raw_json,
                updated_at = EXCLUDED.updated_at
            """,
            (
                payload["ticker"],
                payload["sector"],
                payload["peers"],
                payload["suppliers"],
                payload["customers"],
                payload["thematic_proxies"],
                payload["commodities"],
                payload["rate_sensitive"],
                payload["fx_sensitive"],
                payload["duration_bucket"],
                payload["notes"],
                json.dumps(payload["raw_json"], default=str),
                datetime.now(timezone.utc),
            ),
        )


def upsert_ticker_kg_nodes(
    nodes: list[TickerNode] | None = None,
    *,
    database_url: str | None = None,
    preserve_overlays: bool = True,
) -> int:
    items = nodes if nodes is not None else list(ENTITY_GRAPH.values())
    for node in items:
        upsert_ticker_kg_node(
            node,
            database_url=database_url,
            preserve_overlays=preserve_overlays,
        )
    return len(items)


def get_ticker_kg_node(
    ticker: str,
    *,
    database_url: str | None = None,
) -> dict | None:
    row = fetch_ticker_kg_row(ticker, database_url=database_url)
    if row is not None:
        return row
    static = get_node(ticker)
    return static.to_dict() if static else None
