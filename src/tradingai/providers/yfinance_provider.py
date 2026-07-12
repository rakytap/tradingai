import time
from typing import Any

import pandas as pd
import yfinance as yf

from tradingai.providers.base import MarketDataProvider, Quote

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 60


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No OHLCV data returned for the requested symbol and range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    column_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    missing = [name for name in column_map if name not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns: {', '.join(missing)}")

    normalized = df[list(column_map.keys())].rename(columns=column_map).copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    return normalized


class YFinanceProvider(MarketDataProvider):
    def get_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        cache_key = f"ohlcv:{symbol}:{period}:{interval}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached.copy()

        ticker = yf.Ticker(symbol)
        raw = ticker.history(period=period, interval=interval, auto_adjust=True)
        normalized = _normalize_ohlcv(raw)
        _cache_set(cache_key, normalized)
        return normalized.copy()

    def get_quote(self, symbol: str) -> Quote:
        cache_key = f"quote:{symbol}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = getattr(info, "last_price", None)
        if price is None:
            history = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if history.empty:
                raise ValueError(f"No quote data available for {symbol}.")
            price = float(history["Close"].iloc[-1])

        quote = Quote(
            symbol=symbol.upper(),
            price=float(price),
            currency=getattr(info, "currency", None),
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
        )
        _cache_set(cache_key, quote)
        return quote
