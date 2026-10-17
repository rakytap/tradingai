from __future__ import annotations

from tradingai.analysis.ml.ranking import bullish_sectors_from_regime
from tradingai.data.etf_symbols import is_etf
from tradingai.data.repositories.stocks import list_tickers, list_tickers_by_sector
from tradingai.intraday.config import get_watchlist_cap
from tradingai.regime.sector_proxies import normalize_sector_label


def bullish_sector_watchlist(
    *,
    tickers: list[str] | None = None,
    cap: int | None = None,
) -> list[str]:
    """Equities in L4 bullish sectors, capped for IBKR historical pacing."""
    limit = cap if cap is not None else get_watchlist_cap()
    if tickers:
        cleaned = [t.strip().upper() for t in tickers if t and str(t).strip()]
        equities = [t for t in cleaned if not is_etf(t)]
        return equities[:limit]

    bullish = {
        normalize_sector_label(sector) or sector
        for sector in bullish_sectors_from_regime()
    }
    if not bullish:
        # Fall back to registered non-ETF tickers if board empty
        return [t for t in list_tickers() if not is_etf(t)][:limit]

    selected: list[str] = []
    seen: set[str] = set()
    for sector in sorted(bullish):
        for ticker in list_tickers_by_sector(sector, exclude_etf=True):
            key = ticker.upper()
            if key in seen or is_etf(key):
                continue
            seen.add(key)
            selected.append(key)
            if len(selected) >= limit:
                return selected
    return selected
