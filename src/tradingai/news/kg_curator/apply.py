from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.ticker_kg import (
    fetch_ticker_kg_row,
    upsert_ticker_kg_node,
)
from tradingai.news.entity_graph import get_node, node_from_dict, resolve_ticker_node
from tradingai.news.kg_curator.config import get_kg_proposal_dir
from tradingai.news.kg_curator.validate import validate_proposed_relations


def load_proposal(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "ticker" not in payload:
        raise ValueError("Proposal JSON must include ticker and proposed relations")
    return payload


def latest_pending_proposal(ticker: str, *, proposal_dir: Path | None = None) -> Path | None:
    directory = proposal_dir or get_kg_proposal_dir()
    if not directory.exists():
        return None
    symbol = ticker.upper()
    candidates = sorted(
        directory.glob(f"{symbol}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status", "pending") == "pending":
            return path
    return None


def apply_proposal(
    path: str | Path | None = None,
    *,
    ticker: str | None = None,
    proposal_dir: Path | None = None,
) -> dict:
    """Validate and upsert proposal into ticker_kg_nodes (explicit human apply)."""
    if path is None:
        if not ticker:
            raise ValueError("Provide proposal path or ticker to locate latest pending draft")
        found = latest_pending_proposal(ticker, proposal_dir=proposal_dir)
        if found is None:
            raise FileNotFoundError(f"No pending proposal for {ticker.upper()}")
        path = found

    proposal_path = Path(path)
    proposal = load_proposal(proposal_path)
    symbol = str(proposal["ticker"]).upper()
    mode = str(proposal.get("mode") or "merge")
    existing = resolve_ticker_node(symbol)
    existing_dict = existing.to_dict() if existing else None
    validated = validate_proposed_relations(
        symbol,
        proposal.get("proposed") or {},
        existing=existing_dict if mode == "merge" else None,
        mode=mode,
    )
    base = existing_dict or (get_node(symbol).to_dict() if get_node(symbol) else {"ticker": symbol})
    merged = {
        **base,
        "ticker": symbol,
        **validated["proposed"],
        "notes": (
            f"llm_curator:{proposal.get('model', 'unknown')} "
            f"applied {datetime.now(timezone.utc).date().isoformat()}"
        ),
    }
    node = node_from_dict(merged)
    raw_json = {
        **node.to_dict(),
        "source": "llm_curator",
        "model": proposal.get("model"),
        "proposal_path": str(proposal_path),
        "rationale": validated.get("rationale") or proposal.get("rationale") or {},
        "mode": validated["mode"],
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    initialize_schema()
    upsert_ticker_kg_node(node, raw_json=raw_json, preserve_overlays=False)

    proposal["status"] = "applied"
    proposal["applied_at"] = raw_json["applied_at"]
    proposal["proposed"] = validated["proposed"]
    proposal_path.write_text(json.dumps(proposal, indent=2, default=str))

    stored = fetch_ticker_kg_row(symbol)
    return {
        "ticker": symbol,
        "path": str(proposal_path),
        "status": "applied",
        "node": stored or node.to_dict(),
        "mode": validated["mode"],
    }
