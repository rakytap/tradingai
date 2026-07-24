from __future__ import annotations

import json


SYSTEM_PROMPT = """You are a careful equity research assistant updating a knowledge-graph of tickers.
Return ONLY valid JSON (no markdown) with this schema:
{
  "peers": ["TICKER", ...],
  "suppliers": ["TICKER", ...],
  "customers": ["TICKER", ...],
  "thematic_proxies": ["ETF", ...],
  "rationale": {"TICKER": "short reason", ...}
}
Rules:
- Use US-listed ticker symbols only (1-5 letters), uppercase.
- thematic_proxies should be ETFs (e.g. SMH, SOXX, XLK, XLU, XLI) when relevant.
- Do not include the subject ticker in any list.
- Prefer high-confidence, well-known relationships; omit speculative edges.
- Keep each list to at most 12 symbols.
- rationale keys must be tickers that appear in the lists.
"""


def build_user_prompt(evidence: dict) -> str:
    payload = {
        "subject_ticker": evidence.get("ticker"),
        "stock_metadata": evidence.get("stock"),
        "existing_graph_node": evidence.get("existing_node"),
        "recent_headlines": evidence.get("headlines"),
        "filing_excerpts": evidence.get("filing_excerpts"),
        "task": (
            "Propose peers, suppliers, customers, and thematic_proxies for the subject. "
            "Merge thoughtfully with existing_graph_node when present."
        ),
    }
    return json.dumps(payload, indent=2, default=str)
