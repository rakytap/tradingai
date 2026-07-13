from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf

from tradingai.data.etf_symbols import is_etf

ETF_SECTOR = "ETF"
ETF_INDUSTRY = "Exchange Traded Fund"


@dataclass(frozen=True)
class StockMetadata:
    sector: str | None
    industry: str | None
    market_cap: int | None


def resolve_stock_metadata(ticker: str) -> StockMetadata:
    """Resolve stock metadata without calling yfinance fundamentals for ETFs."""
    symbol = ticker.strip().upper()
    if is_etf(symbol):
        return StockMetadata(sector=ETF_SECTOR, industry=ETF_INDUSTRY, market_cap=None)

    info = yf.Ticker(symbol).info
    market_cap = info.get("marketCap")
    return StockMetadata(
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=int(market_cap) if market_cap is not None else None,
    )
