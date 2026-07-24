from __future__ import annotations

from tradingai.data.repositories.stocks import get_stock
from tradingai.news.entity_graph import resolve_ticker_node


def gather_evidence(ticker: str, *, headline_limit: int = 12) -> dict:
    """Collect context for the LLM curator (deterministic; no LLM call)."""
    symbol = ticker.upper()
    node = resolve_ticker_node(symbol)
    stock = None
    try:
        stock = get_stock(symbol)
    except Exception:
        stock = None

    headlines: list[str] = []
    filing_excerpts: list[str] = []
    try:
        from tradingai.news.tools.news import fetch_company_news, fetch_sec_filings

        news = fetch_company_news(symbol, count=headline_limit)
        for item in news.get("items") or news.get("news") or []:
            if isinstance(item, dict):
                title = item.get("title") or item.get("headline")
                if title:
                    headlines.append(str(title)[:200])
            elif hasattr(item, "title"):
                headlines.append(str(item.title)[:200])
        filings = fetch_sec_filings(symbol, count=3)
        for item in filings.get("items") or filings.get("filings") or []:
            if isinstance(item, dict):
                excerpt = item.get("excerpt") or item.get("title")
                if excerpt:
                    filing_excerpts.append(str(excerpt)[:400])
    except Exception:
        pass

    existing = node.to_dict() if node else None
    digest = []
    if stock:
        digest.append(
            f"sector={stock.get('sector')} industry={stock.get('industry')}"
        )
    if existing:
        digest.append(
            "existing peers="
            + ",".join(existing.get("peers") or [])
            + "; suppliers="
            + ",".join(existing.get("suppliers") or [])
        )
    digest.extend(headlines[:8])
    digest.extend(filing_excerpts[:2])

    return {
        "ticker": symbol,
        "stock": stock,
        "existing_node": existing,
        "headlines": headlines[:headline_limit],
        "filing_excerpts": filing_excerpts[:3],
        "evidence_digest": digest[:20],
    }
