from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tradingai.news.analysis.events import NewsEvent
from tradingai.news.entity_graph import (
    ENTITY_GRAPH,
    RelationKind,
    get_node,
    neighbors,
    nodes_by_commodity,
    nodes_by_flag,
)

RELATION_WEIGHTS: dict[RelationKind, float] = {
    "supplier": 0.8,
    "customer": 0.8,
    "peer": 0.6,
    "thematic_proxy": 0.9,
}
FLAG_WEIGHT = 0.4
COMMODITY_WEIGHT = 0.7

# Theme → which graph expansions to run.
THEME_EXPANSIONS: dict[str, dict] = {
    "ai_infra": {
        "relations": ("supplier", "peer", "thematic_proxy"),
        "commodities": ("power",),
    },
    "semiconductors": {
        "relations": ("supplier", "peer", "thematic_proxy", "customer"),
        "commodities": (),
    },
    "power_equipment": {
        "relations": ("supplier", "customer", "thematic_proxy"),
        "commodities": ("power",),
    },
    "rates_easing": {
        "relations": (),
        "commodities": (),
        "rate_sensitive": True,
    },
    "geopolitics": {
        "relations": ("supplier", "customer"),
        "commodities": (),
        "fx_sensitive": True,
    },
}


@dataclass
class RelatedCandidate:
    ticker: str
    relation: str
    weight: float
    reason: str
    themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _themes_from_input(
    events: list[NewsEvent] | list[dict] | None,
    themes: list[str] | None,
) -> list[str]:
    collected: list[str] = []
    if themes:
        collected.extend(themes)
    for event in events or []:
        if isinstance(event, NewsEvent):
            collected.extend(event.themes)
        elif isinstance(event, dict):
            collected.extend(event.get("themes") or [])
    # Preserve order, unique.
    seen: set[str] = set()
    ordered: list[str] = []
    for theme in collected:
        if theme not in seen:
            seen.add(theme)
            ordered.append(theme)
    return ordered


def _event_type_hint(
    events: list[NewsEvent] | list[dict] | None,
) -> str | None:
    for event in events or []:
        if isinstance(event, NewsEvent):
            return event.event_type
        if isinstance(event, dict) and event.get("event_type"):
            return str(event["event_type"])
    return None


def resolve_related_candidates(
    ticker: str,
    *,
    events: list[NewsEvent] | list[dict] | None = None,
    themes: list[str] | None = None,
    limit: int = 12,
) -> list[dict]:
    """Expand L2 themes through the static L3 entity graph into related tickers."""
    symbol = ticker.upper()
    node = get_node(symbol)
    if node is None:
        return []

    theme_list = _themes_from_input(events, themes)
    if not theme_list:
        # No themes: still return one-hop suppliers + thematic proxies lightly.
        theme_list = ["_default"]

    event_hint = _event_type_hint(events)
    best: dict[str, RelatedCandidate] = {}

    def consider(
        related: str,
        relation: str,
        weight: float,
        theme: str,
        reason: str,
    ) -> None:
        related_u = related.upper()
        if related_u == symbol:
            return
        weight = round(max(0.0, min(1.0, weight)), 4)
        existing = best.get(related_u)
        if existing is not None and existing.weight >= weight:
            if theme not in existing.themes:
                existing.themes.append(theme)
            return
        themes_acc = list(existing.themes) if existing else []
        if theme not in themes_acc and theme != "_default":
            themes_acc.append(theme)
        best[related_u] = RelatedCandidate(
            ticker=related_u,
            relation=relation,
            weight=weight,
            reason=reason,
            themes=themes_acc,
        )

    for theme in theme_list:
        if theme == "_default":
            for related, kind in neighbors(symbol, kinds=("supplier", "thematic_proxy")):
                prior = RELATION_WEIGHTS[kind] * 0.5
                hint = event_hint or "graph"
                consider(
                    related,
                    kind,
                    prior,
                    theme,
                    f"{symbol} {hint} → {kind} {related}",
                )
            continue

        spec = THEME_EXPANSIONS.get(theme)
        if spec is None:
            continue

        for kind in spec.get("relations") or ():
            prior = RELATION_WEIGHTS[kind]
            for related, rel in neighbors(symbol, kinds=(kind,)):
                hint = event_hint or theme
                consider(
                    related,
                    rel,
                    prior,
                    theme,
                    f"{symbol} {hint} → {rel} {related}",
                )
                # Prefer SMH/SOXX/XLU wording for thematic proxies.
                if rel == "thematic_proxy":
                    consider(
                        related,
                        rel,
                        prior,
                        theme,
                        f"{theme} → thematic_proxy {related}",
                    )

        for commodity in spec.get("commodities") or ():
            for commodity_node in nodes_by_commodity(commodity):
                if commodity_node.ticker == symbol:
                    continue
                # Prefer nodes already linked, else commodity cohort.
                linked = {t for t, _ in neighbors(symbol)}
                weight = COMMODITY_WEIGHT if commodity_node.ticker in linked else COMMODITY_WEIGHT * 0.85
                relation = "commodity"
                if commodity_node.ticker in linked:
                    # Keep stronger relation label if also a direct edge.
                    for related, kind in neighbors(symbol):
                        if related == commodity_node.ticker:
                            relation = kind
                            weight = max(weight, RELATION_WEIGHTS[kind])
                            break
                consider(
                    commodity_node.ticker,
                    relation,
                    weight,
                    theme,
                    f"{theme} → commodity:{commodity} {commodity_node.ticker}",
                )

        if spec.get("rate_sensitive"):
            for flag_node in nodes_by_flag(rate_sensitive=True):
                consider(
                    flag_node.ticker,
                    "rate_sensitive",
                    FLAG_WEIGHT,
                    theme,
                    f"{theme} → rate_sensitive {flag_node.ticker}",
                )

        if spec.get("fx_sensitive"):
            for flag_node in nodes_by_flag(fx_sensitive=True):
                consider(
                    flag_node.ticker,
                    "fx_sensitive",
                    FLAG_WEIGHT,
                    theme,
                    f"{theme} → fx_sensitive {flag_node.ticker}",
                )
            for related, kind in neighbors(symbol, kinds=("supplier", "customer")):
                consider(
                    related,
                    kind,
                    RELATION_WEIGHTS[kind],
                    theme,
                    f"{symbol} {theme} → {kind} {related}",
                )

    ranked = sorted(best.values(), key=lambda item: (-item.weight, item.ticker))
    return [item.to_dict() for item in ranked[:limit]]


def ticker_graph_view(ticker: str) -> dict:
    """One-hop inspection payload for MCP."""
    node = get_node(ticker)
    if node is None:
        return {
            "ticker": ticker.upper(),
            "in_graph": False,
            "node": None,
            "neighbors": [],
            "graph_size": len(ENTITY_GRAPH),
        }
    hop = [
        {"ticker": related, "relation": kind}
        for related, kind in neighbors(ticker)
    ]
    return {
        "ticker": node.ticker,
        "in_graph": True,
        "node": node.to_dict(),
        "neighbors": hop,
        "graph_size": len(ENTITY_GRAPH),
    }
