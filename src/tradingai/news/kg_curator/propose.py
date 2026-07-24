from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tradingai.news.kg_curator.config import get_kg_proposal_dir, get_llm_provider, resolve_llm_settings
from tradingai.news.kg_curator.evidence import gather_evidence
from tradingai.news.kg_curator.llm_client import LLMCuratorError, chat_completion_json
from tradingai.news.kg_curator.validate import validate_proposed_relations


def propose_ticker_graph_edges(
    ticker: str,
    *,
    mode: str = "merge",
    model: str | None = None,
    provider: str | None = None,
    proposal_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Call LLM, validate, and write a pending proposal JSON for human review."""
    evidence = gather_evidence(ticker)
    settings = resolve_llm_settings(provider=provider, model=model)
    raw, used_model = chat_completion_json(
        evidence,
        model=model,
        provider=provider,
    )
    validated = validate_proposed_relations(
        evidence["ticker"],
        raw,
        existing=evidence.get("existing_node"),
        mode=mode,
    )
    proposal = {
        "ticker": evidence["ticker"],
        "mode": validated["mode"],
        "proposed": validated["proposed"],
        "rationale": validated["rationale"],
        "evidence_digest": evidence.get("evidence_digest") or [],
        "provider": settings["provider"] or get_llm_provider(),
        "model": used_model or settings["model"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        proposal["path"] = None
        return proposal

    out_dir = proposal_dir or get_kg_proposal_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{evidence['ticker']}_{stamp}.json"
    path.write_text(json.dumps(proposal, indent=2, default=str))
    proposal["path"] = str(path)
    return proposal


__all__ = ["propose_ticker_graph_edges", "LLMCuratorError"]
