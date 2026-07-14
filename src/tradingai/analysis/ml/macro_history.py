from __future__ import annotations

import pandas as pd

from tradingai.analysis.ml.config import (
    MIN_MACRO_HISTORY_ROWS,
    get_macro_sync_tickers,
    yfinance_macro_symbol,
)
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.regime.analysis.historical import _normalize_index

REQUIRED_MACRO_TICKERS = ("SPY", "VIX", "HYG", "LQD")
OPTIONAL_MACRO_TICKERS = ("TNX",)


def _normalize_close_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    normalized = series.copy()
    normalized.index = _normalize_index(normalized.index)
    return normalized


def _close_series_from_db(ticker: str, *, database_url: str | None = None) -> pd.Series:
    frame = load_ticker_ohlcv(ticker, database_url=database_url)
    if frame.empty:
        return pd.Series(dtype=float)
    return _normalize_close_series(frame["close"])


def _fetch_yfinance_close(ticker: str, *, period: str = "2y") -> pd.Series:
    from tradingai.regime.providers.yfinance_macro_provider import YFinanceMacroProvider

    provider = YFinanceMacroProvider()
    symbol = yfinance_macro_symbol(ticker)
    return provider._history(symbol, period=period)


def _load_or_fetch_series(
    ticker: str,
    *,
    database_url: str | None = None,
    period: str = "2y",
) -> pd.Series:
    series = _close_series_from_db(ticker, database_url=database_url)
    if len(series) >= MIN_MACRO_HISTORY_ROWS:
        return series
    fetched = _fetch_yfinance_close(ticker, period=period)
    if fetched.empty:
        return series
    return _normalize_close_series(fetched)


def load_macro_histories(
    *,
    database_url: str | None = None,
    period: str = "2y",
) -> dict[str, pd.Series]:
    configured = get_macro_sync_tickers()
    tickers = list(dict.fromkeys([*REQUIRED_MACRO_TICKERS, *configured, *OPTIONAL_MACRO_TICKERS]))

    histories: dict[str, pd.Series] = {}
    for ticker in tickers:
        if ticker in histories:
            continue
        try:
            series = _load_or_fetch_series(ticker, database_url=database_url, period=period)
        except Exception:
            if ticker in OPTIONAL_MACRO_TICKERS:
                continue
            raise
        if not series.empty:
            histories[ticker.upper()] = series

    missing = [ticker for ticker in REQUIRED_MACRO_TICKERS if ticker not in histories]
    if missing:
        raise ValueError(
            f"Missing macro price history for {', '.join(missing)}. "
            "Run tradingai-sync-prices for macro tickers or ensure network access for yfinance fallback."
        )
    return histories
