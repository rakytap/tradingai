from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from tradingai.providers.base import MarketDataProvider, Quote

ET = ZoneInfo("America/New_York")

_PERIOD_RE = re.compile(r"^(\d+)([dDwWmMyY])$")

_INTERVAL_MAP: dict[str, TimeFrame] = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "1min": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "5min": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "15min": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame.Day,
    "1day": TimeFrame.Day,
}


def _require_credentials() -> tuple[str, str]:
    key = (os.getenv("ALPACA_API_KEY") or "").strip()
    secret = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )
    return key, secret


@lru_cache
def _client() -> StockHistoricalDataClient:
    key, secret = _require_credentials()
    return StockHistoricalDataClient(key, secret)


def interval_to_timeframe(interval: str) -> TimeFrame:
    key = (interval or "").strip().lower()
    if key not in _INTERVAL_MAP:
        raise ValueError(
            f"Unsupported Alpaca interval '{interval}'. "
            f"Supported: {', '.join(sorted(_INTERVAL_MAP))}."
        )
    return _INTERVAL_MAP[key]


def period_to_start_end(period: str) -> tuple[datetime, datetime]:
    """Map a yfinance-style period string to an inclusive UTC [start, end] window."""
    raw = (period or "").strip()
    match = _PERIOD_RE.fullmatch(raw)
    if match is None:
        raise ValueError(
            f"Unsupported Alpaca period '{period}'. Use forms like 5d, 60d, 1mo, 1y."
        )
    amount = int(match.group(1))
    unit = match.group(2).lower()
    end = datetime.now(timezone.utc)
    if unit == "d":
        delta = timedelta(days=amount)
    elif unit == "w":
        delta = timedelta(weeks=amount)
    elif unit == "m":
        delta = timedelta(days=amount * 31)
    else:  # y
        delta = timedelta(days=amount * 365)
    return end - delta, end


def bars_df_to_ohlcv(frame: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Normalize Alpaca bars DataFrame to open/high/low/close/volume with DatetimeIndex."""
    if frame is None or frame.empty:
        raise ValueError("No OHLCV data returned for the requested symbol and range.")

    work = frame.copy()
    if isinstance(work.index, pd.MultiIndex):
        # Index levels are typically (symbol, timestamp).
        names = list(work.index.names)
        if "symbol" in names and symbol is not None:
            work = work.xs(symbol.upper(), level="symbol")
        elif names and names[0] == "symbol":
            work = work.droplevel(0)
        else:
            work = work.reset_index(level=0, drop=True)

    column_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    rename = {col: column_map[col] for col in work.columns if col in column_map}
    work = work.rename(columns=rename)
    missing = [name for name in ("open", "high", "low", "close") if name not in work.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns: {', '.join(missing)}")
    if "volume" not in work.columns:
        work["volume"] = 0.0

    if not isinstance(work.index, pd.DatetimeIndex):
        if "timestamp" in work.columns:
            work = work.set_index("timestamp")
        else:
            raise ValueError("Alpaca bars require a DatetimeIndex.")
    work.index = pd.to_datetime(work.index)
    if work.index.tz is None:
        work.index = work.index.tz_localize(timezone.utc).tz_convert(ET)
    else:
        work.index = work.index.tz_convert(ET)
    work = work.sort_index()
    return work[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_stock_bars(
    symbol: str,
    *,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime | None = None,
    feed: DataFeed | str | None = None,
) -> pd.DataFrame:
    """Fetch bars for one symbol and return normalized OHLCV."""
    client = _client()
    kwargs: dict = {
        "symbol_or_symbols": symbol.upper(),
        "timeframe": timeframe,
        "start": start,
    }
    if end is not None:
        kwargs["end"] = end
    if feed is not None:
        kwargs["feed"] = DataFeed(feed) if isinstance(feed, str) else feed
    request = StockBarsRequest(**kwargs)
    raw = client.get_stock_bars(request).df
    return bars_df_to_ohlcv(raw, symbol=symbol)


class AlpacaProvider(MarketDataProvider):
    """Alpaca market-data provider (historical bars + latest quote)."""

    def get_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        start, end = period_to_start_end(period)
        timeframe = interval_to_timeframe(interval)
        return fetch_stock_bars(
            symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    def get_quote(self, symbol: str) -> Quote:
        client = _client()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
        quotes = client.get_stock_latest_quote(request)
        quote = quotes.get(symbol.upper()) if isinstance(quotes, dict) else None
        if quote is None and hasattr(quotes, "__getitem__"):
            try:
                quote = quotes[symbol.upper()]
            except Exception:
                quote = None
        if quote is None:
            raise ValueError(f"No quote data available for {symbol}.")
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        price = None
        if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
            price = (float(bid) + float(ask)) / 2.0
        elif ask is not None and float(ask) > 0:
            price = float(ask)
        elif bid is not None and float(bid) > 0:
            price = float(bid)
        if price is None or price <= 0:
            raise ValueError(f"No usable quote price for {symbol}.")
        ts = getattr(quote, "timestamp", None)
        if ts is not None and getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return Quote(
            symbol=symbol.upper(),
            price=float(price),
            currency="USD",
            timestamp=ts,
        )
