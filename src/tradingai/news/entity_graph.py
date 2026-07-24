from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

RelationKind = Literal["peer", "supplier", "customer", "thematic_proxy"]
DurationBucket = Literal["short", "medium", "long"]


@dataclass(frozen=True)
class TickerNode:
    ticker: str
    sector: str | None = None
    peers: tuple[str, ...] = ()
    suppliers: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()
    thematic_proxies: tuple[str, ...] = ()
    commodities: tuple[str, ...] = ()
    rate_sensitive: bool = False
    fx_sensitive: bool = False
    duration_bucket: DurationBucket | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ticker"] = self.ticker.upper()
        payload["peers"] = list(self.peers)
        payload["suppliers"] = list(self.suppliers)
        payload["customers"] = list(self.customers)
        payload["thematic_proxies"] = list(self.thematic_proxies)
        payload["commodities"] = list(self.commodities)
        return payload


def _node(
    ticker: str,
    *,
    sector: str | None = None,
    peers: Iterable[str] = (),
    suppliers: Iterable[str] = (),
    customers: Iterable[str] = (),
    thematic_proxies: Iterable[str] = (),
    commodities: Iterable[str] = (),
    rate_sensitive: bool = False,
    fx_sensitive: bool = False,
    duration_bucket: DurationBucket | None = None,
    notes: str | None = None,
) -> TickerNode:
    return TickerNode(
        ticker=ticker.upper(),
        sector=sector,
        peers=tuple(t.upper() for t in peers),
        suppliers=tuple(t.upper() for t in suppliers),
        customers=tuple(t.upper() for t in customers),
        thematic_proxies=tuple(t.upper() for t in thematic_proxies),
        commodities=tuple(c.lower() for c in commodities),
        rate_sensitive=rate_sensitive,
        fx_sensitive=fx_sensitive,
        duration_bucket=duration_bucket,
        notes=notes,
    )


# Curated static L3 graph — source of truth (not scraped).
ENTITY_GRAPH: dict[str, TickerNode] = {
    # Hyperscalers
    "MSFT": _node(
        "MSFT",
        sector="Technology",
        peers=("AAPL", "GOOGL", "AMZN", "META"),
        suppliers=("NVDA", "AVGO", "AMD", "TSM"),
        thematic_proxies=("SMH", "XLK", "XLU"),
        commodities=("power",),
        notes="AI capex / cloud; power + semis exposure",
    ),
    "AMZN": _node(
        "AMZN",
        sector="Consumer Cyclical",
        peers=("MSFT", "GOOGL", "META"),
        suppliers=("NVDA", "AVGO", "AMD"),
        thematic_proxies=("SMH", "XLK", "XLU"),
        commodities=("power",),
    ),
    "GOOGL": _node(
        "GOOGL",
        sector="Communication Services",
        peers=("MSFT", "META", "AMZN"),
        suppliers=("NVDA", "AVGO", "TSM"),
        thematic_proxies=("SMH", "XLK", "XLU"),
        commodities=("power",),
    ),
    "META": _node(
        "META",
        sector="Communication Services",
        peers=("GOOGL", "MSFT", "AMZN"),
        suppliers=("NVDA", "AVGO", "TSM"),
        thematic_proxies=("SMH", "XLK"),
        commodities=("power",),
    ),
    "AAPL": _node(
        "AAPL",
        sector="Technology",
        peers=("MSFT", "GOOGL"),
        suppliers=("TSM", "AVGO", "QCOM"),
        thematic_proxies=("XLK", "SMH"),
        fx_sensitive=True,
    ),
    # Semis / AI infra
    "NVDA": _node(
        "NVDA",
        sector="Technology",
        peers=("AMD", "AVGO", "TSM"),
        suppliers=("TSM", "ASML", "AMAT"),
        customers=("MSFT", "AMZN", "GOOGL", "META"),
        thematic_proxies=("SMH", "SOXX", "XLK"),
        commodities=("power",),
        fx_sensitive=True,
        notes="GPU AI infra",
    ),
    "AMD": _node(
        "AMD",
        sector="Technology",
        peers=("NVDA", "INTC", "AVGO"),
        suppliers=("TSM", "ASML"),
        customers=("MSFT", "AMZN", "GOOGL"),
        thematic_proxies=("SMH", "SOXX", "XLK"),
        fx_sensitive=True,
    ),
    "AVGO": _node(
        "AVGO",
        sector="Technology",
        peers=("NVDA", "QCOM", "MRVL"),
        suppliers=("TSM", "ASML"),
        customers=("MSFT", "AAPL", "GOOGL"),
        thematic_proxies=("SMH", "SOXX", "XLK"),
        fx_sensitive=True,
    ),
    "TSM": _node(
        "TSM",
        sector="Technology",
        peers=("INTC",),
        suppliers=("ASML", "AMAT", "LRCX"),
        customers=("NVDA", "AMD", "AAPL", "AVGO"),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
        notes="Foundry; geopolitics / FX sensitive",
    ),
    "ASML": _node(
        "ASML",
        sector="Technology",
        peers=("AMAT", "LRCX", "KLAC"),
        customers=("TSM", "INTC"),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
    ),
    "AMAT": _node(
        "AMAT",
        sector="Technology",
        peers=("ASML", "LRCX", "KLAC"),
        customers=("TSM", "INTC"),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
    ),
    "LRCX": _node(
        "LRCX",
        sector="Technology",
        peers=("AMAT", "KLAC", "ASML"),
        customers=("TSM", "INTC"),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
    ),
    "QCOM": _node(
        "QCOM",
        sector="Technology",
        peers=("AVGO", "MRVL", "AAPL"),
        suppliers=("TSM",),
        customers=("AAPL",),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
    ),
    "INTC": _node(
        "INTC",
        sector="Technology",
        peers=("AMD", "TSM", "NVDA"),
        suppliers=("ASML", "AMAT"),
        thematic_proxies=("SMH", "SOXX"),
        fx_sensitive=True,
    ),
    "MRVL": _node(
        "MRVL",
        sector="Technology",
        peers=("AVGO", "QCOM"),
        customers=("MSFT", "AMZN"),
        thematic_proxies=("SMH", "SOXX"),
    ),
    # Power / grid / utilities
    "VRT": _node(
        "VRT",
        sector="Industrials",
        peers=("ETN", "HUBB"),
        customers=("MSFT", "AMZN", "GOOGL", "META"),
        thematic_proxies=("XLU", "XLI"),
        commodities=("power",),
        notes="Data-center power / cooling",
    ),
    "ETN": _node(
        "ETN",
        sector="Industrials",
        peers=("VRT", "HUBB", "EMR"),
        customers=("MSFT", "AMZN"),
        thematic_proxies=("XLU", "XLI"),
        commodities=("power",),
        notes="Electrical / grid equipment",
    ),
    "CEG": _node(
        "CEG",
        sector="Utilities",
        peers=("NEE", "VST"),
        customers=("MSFT", "AMZN", "GOOGL"),
        thematic_proxies=("XLU",),
        commodities=("power",),
        rate_sensitive=True,
        duration_bucket="long",
        notes="Nuclear / power for AI load",
    ),
    "NEE": _node(
        "NEE",
        sector="Utilities",
        peers=("CEG", "DUK", "SO"),
        thematic_proxies=("XLU",),
        commodities=("power",),
        rate_sensitive=True,
        duration_bucket="long",
    ),
    "VST": _node(
        "VST",
        sector="Utilities",
        peers=("CEG", "NEE"),
        customers=("MSFT", "AMZN"),
        thematic_proxies=("XLU",),
        commodities=("power",),
        rate_sensitive=True,
        duration_bucket="medium",
    ),
    "HUBB": _node(
        "HUBB",
        sector="Industrials",
        peers=("ETN", "VRT"),
        thematic_proxies=("XLI", "XLU"),
        commodities=("power",),
    ),
    # Rate / duration / FX proxies
    "PLD": _node(
        "PLD",
        sector="Real Estate",
        peers=("EQIX", "AMT"),
        thematic_proxies=("XLRE",),
        rate_sensitive=True,
        duration_bucket="long",
    ),
    "EQIX": _node(
        "EQIX",
        sector="Real Estate",
        peers=("PLD", "DLR"),
        customers=("MSFT", "AMZN", "GOOGL"),
        thematic_proxies=("XLRE", "XLK"),
        commodities=("power",),
        rate_sensitive=True,
        duration_bucket="long",
        notes="Data-center REIT",
    ),
    "JPM": _node(
        "JPM",
        sector="Financial Services",
        peers=("BAC", "GS"),
        thematic_proxies=("XLF",),
        rate_sensitive=True,
        duration_bucket="short",
        fx_sensitive=True,
    ),
    "XOM": _node(
        "XOM",
        sector="Energy",
        peers=("CVX", "COP"),
        thematic_proxies=("XLE",),
        commodities=("oil",),
        fx_sensitive=True,
    ),
    # Thematic ETFs
    "SMH": _node(
        "SMH",
        sector="ETF",
        peers=("SOXX", "XLK"),
        notes="VanEck Semiconductor ETF",
    ),
    "SOXX": _node(
        "SOXX",
        sector="ETF",
        peers=("SMH", "XLK"),
        notes="iShares Semiconductor ETF",
    ),
    "XLU": _node(
        "XLU",
        sector="ETF",
        peers=("XLI",),
        commodities=("power",),
        rate_sensitive=True,
        duration_bucket="long",
        notes="Utilities Select Sector SPDR",
    ),
    "XLK": _node(
        "XLK",
        sector="ETF",
        peers=("SMH", "SOXX"),
        notes="Technology Select Sector SPDR",
    ),
    "XLI": _node(
        "XLI",
        sector="ETF",
        peers=("XLU",),
        notes="Industrials Select Sector SPDR",
    ),
    "XLRE": _node(
        "XLRE",
        sector="ETF",
        rate_sensitive=True,
        duration_bucket="long",
    ),
    "XLF": _node(
        "XLF",
        sector="ETF",
        rate_sensitive=True,
        duration_bucket="short",
    ),
    "XLE": _node(
        "XLE",
        sector="ETF",
        commodities=("oil",),
        fx_sensitive=True,
    ),
}


RELATION_ATTRS: dict[RelationKind, str] = {
    "peer": "peers",
    "supplier": "suppliers",
    "customer": "customers",
    "thematic_proxy": "thematic_proxies",
}


def get_node(ticker: str) -> TickerNode | None:
    return ENTITY_GRAPH.get(ticker.upper())


def node_from_dict(data: dict) -> TickerNode:
    """Build a TickerNode from a DB/API dict (relation lists may be empty)."""
    duration = data.get("duration_bucket")
    if duration is not None:
        duration = str(duration)
    return _node(
        str(data.get("ticker", "")),
        sector=data.get("sector"),
        peers=data.get("peers") or (),
        suppliers=data.get("suppliers") or (),
        customers=data.get("customers") or (),
        thematic_proxies=data.get("thematic_proxies") or (),
        commodities=data.get("commodities") or (),
        rate_sensitive=bool(data.get("rate_sensitive", False)),
        fx_sensitive=bool(data.get("fx_sensitive", False)),
        duration_bucket=duration,  # type: ignore[arg-type]
        notes=data.get("notes"),
    )


def resolve_ticker_node(
    ticker: str,
    *,
    database_url: str | None = None,
) -> TickerNode | None:
    """Prefer DB overlay row; fall back to static ENTITY_GRAPH. Never raises."""
    try:
        from tradingai.data.repositories.ticker_kg import fetch_ticker_kg_row

        row = fetch_ticker_kg_row(ticker, database_url=database_url)
        if row is not None:
            return node_from_dict(row)
    except Exception:
        pass
    return get_node(ticker)


def list_tickers() -> list[str]:
    return sorted(ENTITY_GRAPH.keys())


def neighbors(
    ticker: str,
    *,
    kinds: Iterable[RelationKind] | None = None,
) -> list[tuple[str, RelationKind]]:
    node = resolve_ticker_node(ticker)
    if node is None:
        return []
    selected = list(kinds) if kinds is not None else list(RELATION_ATTRS.keys())
    out: list[tuple[str, RelationKind]] = []
    for kind in selected:
        attr = RELATION_ATTRS[kind]
        for related in getattr(node, attr):
            out.append((related, kind))
    return out


def nodes_by_commodity(commodity: str) -> list[TickerNode]:
    key = commodity.lower()
    return [node for node in ENTITY_GRAPH.values() if key in node.commodities]


def nodes_by_flag(
    *,
    rate_sensitive: bool | None = None,
    fx_sensitive: bool | None = None,
) -> list[TickerNode]:
    results: list[TickerNode] = []
    for node in ENTITY_GRAPH.values():
        if rate_sensitive is not None and node.rate_sensitive != rate_sensitive:
            continue
        if fx_sensitive is not None and node.fx_sensitive != fx_sensitive:
            continue
        results.append(node)
    return results
