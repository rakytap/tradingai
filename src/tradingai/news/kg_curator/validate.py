from __future__ import annotations

import re

from tradingai.data.etf_symbols import KNOWN_ETF_TICKERS, is_etf
from tradingai.news.kg_curator.config import MAX_RELATION_LIST

TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
RELATION_KEYS = ("peers", "suppliers", "customers", "thematic_proxies")


def _normalize_symbol(raw: object) -> str | None:
    if raw is None:
        return None
    symbol = str(raw).strip().upper().replace(".", "")
    if not symbol or not TICKER_RE.match(symbol):
        return None
    return symbol


def sanitize_relation_list(
    values: object,
    *,
    subject: str,
    thematic: bool = False,
    limit: int = MAX_RELATION_LIST,
) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        symbol = _normalize_symbol(item)
        if symbol is None or symbol == subject.upper():
            continue
        if thematic and not (symbol in KNOWN_ETF_TICKERS or is_etf(symbol)):
            # Allow unknown ETFs that look like tickers but prefer known; keep if 2-4 letters common ETF shape
            if len(symbol) > 4:
                continue
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
        if len(out) >= limit:
            break
    return out


def validate_proposed_relations(
    subject: str,
    proposed: dict,
    *,
    existing: dict | None = None,
    mode: str = "merge",
) -> dict:
    """Return cleaned relation lists; optionally merge with existing node."""
    symbol = subject.upper()
    cleaned = {
        "peers": sanitize_relation_list(proposed.get("peers"), subject=symbol),
        "suppliers": sanitize_relation_list(proposed.get("suppliers"), subject=symbol),
        "customers": sanitize_relation_list(proposed.get("customers"), subject=symbol),
        "thematic_proxies": sanitize_relation_list(
            proposed.get("thematic_proxies"),
            subject=symbol,
            thematic=True,
        ),
    }
    if mode == "merge" and existing:
        for key in RELATION_KEYS:
            prior = [str(x).upper() for x in (existing.get(key) or []) if x]
            merged: list[str] = []
            seen: set[str] = set()
            for item in [*prior, *cleaned[key]]:
                if item == symbol or item in seen:
                    continue
                if not TICKER_RE.match(item):
                    continue
                seen.add(item)
                merged.append(item)
                if len(merged) >= MAX_RELATION_LIST:
                    break
            cleaned[key] = merged

    rationale_in = proposed.get("rationale") or {}
    rationale: dict[str, str] = {}
    if isinstance(rationale_in, dict):
        allowed = {
            t
            for key in RELATION_KEYS
            for t in cleaned[key]
        }
        for key, value in rationale_in.items():
            symbol_key = _normalize_symbol(key)
            if symbol_key and symbol_key in allowed:
                rationale[symbol_key] = str(value)[:240]
    return {
        "proposed": cleaned,
        "rationale": rationale,
        "mode": mode if mode in {"merge", "replace"} else "merge",
    }
