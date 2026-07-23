from __future__ import annotations

from tradingai.data.repositories.stocks import get_stock

DEFAULT_REGIME_PROXY = "SPY"

# Canonical board sectors (display label → ETF).
SECTOR_BOARD: dict[str, str] = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# yfinance / GICS-style aliases → canonical board label.
_SECTOR_ALIASES: dict[str, str] = {
    "technology": "Technology",
    "energy": "Energy",
    "healthcare": "Healthcare",
    "financial services": "Financial Services",
    "financials": "Financial Services",
    "consumer cyclical": "Consumer Cyclical",
    "consumer discretionary": "Consumer Cyclical",
    "consumer defensive": "Consumer Defensive",
    "consumer staples": "Consumer Defensive",
    "industrials": "Industrials",
    "basic materials": "Basic Materials",
    "materials": "Basic Materials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
    "communication services": "Communication Services",
}

SECTOR_REGIME_PROXIES: dict[str, str] = {
    key: SECTOR_BOARD[label] for key, label in _SECTOR_ALIASES.items()
}
SECTOR_REGIME_PROXIES["etf"] = DEFAULT_REGIME_PROXY

SECTOR_PROXY_TICKERS: tuple[str, ...] = tuple(sorted(set(SECTOR_BOARD.values())))


def normalize_sector_label(sector: str | None) -> str | None:
    if sector is None:
        return None
    cleaned = str(sector).strip()
    if not cleaned:
        return None
    return _SECTOR_ALIASES.get(cleaned.lower(), cleaned)


def sector_regime_proxy(sector: str | None) -> str:
    """Return the ETF used for sector trend regime classification."""
    label = normalize_sector_label(sector)
    if label is None:
        return DEFAULT_REGIME_PROXY
    if label in SECTOR_BOARD:
        return SECTOR_BOARD[label]
    return SECTOR_REGIME_PROXIES.get(str(sector).strip().lower(), DEFAULT_REGIME_PROXY)


def resolve_regime_proxy_for_ticker(
    ticker: str,
    *,
    database_url: str | None = None,
) -> tuple[str | None, str]:
    """Return ``(canonical_sector_or_raw, proxy_ticker)`` for a stock symbol."""
    stock = get_stock(ticker, database_url=database_url)
    sector = None
    if stock and stock.get("sector"):
        sector = str(stock["sector"]).strip() or None
    canonical = normalize_sector_label(sector)
    return canonical or sector, sector_regime_proxy(sector)
